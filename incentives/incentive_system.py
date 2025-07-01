"""
SistemaAvaliacao: sistema de avaliação e pontuação do CogniCode.
"""
import sqlite3
from datetime import datetime

class SistemaAvaliacao:
    MAX_PONTOS_SUPPLY = 100_000_000.0
    def __init__(self, evolution_pipeline):
        self.pipeline = evolution_pipeline
        self.pontos_supply = 0.0
        self._init_db()
    def _init_db(self):
        self.conn = sqlite3.connect('cognicode_avaliacao.db')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS pontuacoes (agente_id TEXT PRIMARY KEY, pontuacao REAL NOT NULL DEFAULT 0.0)''')
        self.conn.commit()
    def get_pontuacao(self, agente_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT pontuacao FROM pontuacoes WHERE agente_id = ?', (agente_id,))
        result = cursor.fetchone()
        return result[0] if result else 0.0
    def registrar_agente(self, agente_id):
        if self.get_pontuacao(agente_id) == 0.0:
            self._update_pontuacao(agente_id, 0.0)
            print(f"Agente {agente_id} registrado no sistema de avaliação.")
        else:
            print(f"Agente {agente_id} já está registrado.")
    def _update_pontuacao(self, agente_id, valor):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO pontuacoes (agente_id, pontuacao) VALUES (?, ?) ON CONFLICT(agente_id) DO UPDATE SET pontuacao = pontuacao + ?''', (agente_id, valor, valor))
        self.conn.commit()
