from __future__ import annotations
import json
import math
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from .models import AgentName, ControllerSession, EventOutcome, MemoryRecord

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum((a * b for a, b in zip(left, right)))
    left_norm = math.sqrt(sum((value * value for value in left)))
    right_norm = math.sqrt(sum((value * value for value in right)))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return -1.0
    return dot / (left_norm * right_norm)

class SQLiteMemoryStore:

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(Path(database_path).expanduser().resolve())
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA foreign_keys=ON')
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            connection.executescript("\n                CREATE TABLE IF NOT EXISTS sessions (\n                    prosumer_id TEXT PRIMARY KEY,\n                    state_json TEXT NOT NULL,\n                    updated_at TEXT NOT NULL\n                );\n\n                CREATE TABLE IF NOT EXISTS orchestration_events (\n                    event_id TEXT PRIMARY KEY,\n                    prosumer_id TEXT NOT NULL,\n                    decision_step INTEGER NOT NULL,\n                    completed_step INTEGER,\n                    status TEXT NOT NULL CHECK(status IN ('pending', 'completed')),\n                    context_json TEXT NOT NULL,\n                    context_text TEXT NOT NULL,\n                    selected_agent TEXT NOT NULL,\n                    trigger_reasons_json TEXT NOT NULL,\n                    outcome_json TEXT,\n                    salient INTEGER NOT NULL DEFAULT 0,\n                    embedding_json TEXT,\n                    embedding_id TEXT,\n                    embedding_dimensions INTEGER,\n                    created_at TEXT NOT NULL,\n                    completed_at TEXT\n                );\n\n                CREATE INDEX IF NOT EXISTS idx_events_recent\n                ON orchestration_events(prosumer_id, status, completed_step DESC);\n\n                CREATE INDEX IF NOT EXISTS idx_events_salient\n                ON orchestration_events(prosumer_id, salient, completed_step DESC);\n                ")
            columns = {row['name'] for row in connection.execute('PRAGMA table_info(orchestration_events)').fetchall()}
            if 'embedding_id' not in columns:
                connection.execute('ALTER TABLE orchestration_events ADD COLUMN embedding_id TEXT')
            if 'embedding_dimensions' not in columns:
                connection.execute('ALTER TABLE orchestration_events ADD COLUMN embedding_dimensions INTEGER')

    def load_session(self, prosumer_id: str, default_agent: AgentName) -> ControllerSession:
        with self._lock, self._connection() as connection:
            row = connection.execute('SELECT state_json FROM sessions WHERE prosumer_id = ?', (prosumer_id,)).fetchone()
        if row is None:
            return ControllerSession(prosumer_id=prosumer_id, active_agent=default_agent)
        return ControllerSession.model_validate_json(row['state_json'])

    def save_session(self, session: ControllerSession) -> None:
        payload = session.model_dump_json()
        with self._lock, self._connection() as connection:
            connection.execute('\n                INSERT INTO sessions(prosumer_id, state_json, updated_at)\n                VALUES (?, ?, ?)\n                ON CONFLICT(prosumer_id) DO UPDATE SET\n                    state_json = excluded.state_json,\n                    updated_at = excluded.updated_at\n                ', (session.prosumer_id, payload, _utc_now()))

    def stage_event(self, *, prosumer_id: str, decision_step: int, context: Dict[str, Any], selected_agent: AgentName, trigger_reasons: Sequence[str]) -> str:
        event_id = uuid.uuid4().hex
        context_json = _canonical_json(context)
        with self._lock, self._connection() as connection:
            connection.execute("\n                INSERT INTO orchestration_events(\n                    event_id, prosumer_id, decision_step, status,\n                    context_json, context_text, selected_agent,\n                    trigger_reasons_json, created_at\n                ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?)\n                ", (event_id, prosumer_id, decision_step, context_json, context_json, selected_agent.value, _canonical_json(list(trigger_reasons)), _utc_now()))
        return event_id

    def pending_event(self, event_id: str) -> Dict[str, Any]:
        with self._lock, self._connection() as connection:
            row = connection.execute('SELECT * FROM orchestration_events WHERE event_id = ?', (event_id,)).fetchone()
        if row is None:
            raise KeyError(f'Unknown event_id: {event_id}')
        return dict(row)

    def pending_events(self, prosumer_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute("\n                SELECT event_id, prosumer_id, decision_step, selected_agent,\n                       trigger_reasons_json, created_at\n                FROM orchestration_events\n                WHERE prosumer_id = ? AND status = 'pending'\n                ORDER BY decision_step\n                ", (prosumer_id,)).fetchall()
        return [{'event_id': row['event_id'], 'prosumer_id': row['prosumer_id'], 'decision_step': int(row['decision_step']), 'selected_agent': row['selected_agent'], 'trigger_reasons': json.loads(row['trigger_reasons_json']), 'created_at': row['created_at']} for row in rows]

    def complete_event(self, *, event_id: str, completed_step: int, outcome: EventOutcome, embedding: Optional[Sequence[float]], embedding_identifier: Optional[str], salience_outcome_threshold: float) -> bool:
        row = self.pending_event(event_id)
        if row['status'] != 'pending':
            raise ValueError(f'Event {event_id} has already been completed.')
        if completed_step <= int(row['decision_step']):
            raise ValueError('completed_step must be later than decision_step.')
        salient = self.will_be_salient(row=row, outcome=outcome, salience_outcome_threshold=salience_outcome_threshold)
        with self._lock, self._connection() as connection:
            connection.execute("\n                UPDATE orchestration_events SET\n                    completed_step = ?, status = 'completed', outcome_json = ?,\n                    salient = ?, embedding_json = ?, embedding_id = ?,\n                    embedding_dimensions = ?, completed_at = ?\n                WHERE event_id = ? AND status = 'pending'\n                ", (completed_step, outcome.model_dump_json(), int(salient), _canonical_json(list(embedding)) if embedding is not None else None, embedding_identifier, len(embedding) if embedding is not None else None, _utc_now(), event_id))
        return salient

    @staticmethod
    def will_be_salient(*, row: Dict[str, Any] | sqlite3.Row, outcome: EventOutcome, salience_outcome_threshold: float) -> bool:
        reasons = json.loads(row['trigger_reasons_json'])
        context = json.loads(row['context_json'])
        instruction = str(context.get('instruction', 'none')).strip().lower()
        explicit_command = instruction not in {'', 'none', 'waiting', 'no command'}
        return explicit_command or 'instruction_change' in reasons or 'grid_alert' in reasons or outcome.constraint_violation or (outcome.metric_improvement.magnitude() >= salience_outcome_threshold)

    @staticmethod
    def _row_to_memory(row: sqlite3.Row, similarity: Optional[float]=None) -> MemoryRecord:
        context = json.loads(row['context_json'])
        compact_context = {'price_regime': context.get('price_regime'), 'trends': context.get('trends'), 'asset_state': context.get('asset_state'), 'pcc_state': context.get('pcc_state'), 'instruction': context.get('instruction'), 'grid_alert': context.get('grid_alert')}
        return MemoryRecord(event_id=row['event_id'], decision_step=int(row['decision_step']), completed_step=int(row['completed_step']), selected_agent=AgentName(row['selected_agent']), trigger_reasons=json.loads(row['trigger_reasons_json']), context_summary=compact_context, outcome=EventOutcome.model_validate_json(row['outcome_json']), similarity=similarity)

    def recent_completed(self, *, prosumer_id: str, before_step: int, limit: int) -> List[MemoryRecord]:
        with self._lock, self._connection() as connection:
            rows = connection.execute("\n                SELECT * FROM orchestration_events\n                WHERE prosumer_id = ? AND status = 'completed'\n                  AND completed_step < ?\n                ORDER BY completed_step DESC\n                LIMIT ?\n                ", (prosumer_id, before_step, limit)).fetchall()
        return [self._row_to_memory(row) for row in reversed(rows)]

    def similar_salient(self, *, prosumer_id: str, before_step: int, query_embedding: Sequence[float], embedding_identifier: str, top_k: int, candidate_limit: int=5000) -> List[MemoryRecord]:
        with self._lock, self._connection() as connection:
            rows = connection.execute("\n                SELECT * FROM orchestration_events\n                WHERE prosumer_id = ? AND status = 'completed'\n                  AND salient = 1 AND completed_step < ?\n                  AND embedding_json IS NOT NULL\n                  AND embedding_id = ? AND embedding_dimensions = ?\n                ORDER BY completed_step DESC\n                LIMIT ?\n                ", (prosumer_id, before_step, embedding_identifier, len(query_embedding), candidate_limit)).fetchall()
        scored: List[tuple[float, sqlite3.Row]] = []
        for row in rows:
            candidate = json.loads(row['embedding_json'])
            similarity = _cosine(query_embedding, candidate)
            scored.append((similarity, row))
        scored.sort(key=lambda item: (item[0], item[1]['completed_step']), reverse=True)
        return [self._row_to_memory(row, score) for score, row in scored[:top_k]]

    def event_counts(self, prosumer_id: str) -> Dict[str, int]:
        with self._lock, self._connection() as connection:
            rows = connection.execute('\n                SELECT status, COUNT(*) AS count\n                FROM orchestration_events\n                WHERE prosumer_id = ? GROUP BY status\n                ', (prosumer_id,)).fetchall()
        result = {'pending': 0, 'completed': 0}
        result.update({row['status']: int(row['count']) for row in rows})
        return result
