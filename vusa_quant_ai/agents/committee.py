"""Runs the DAILY AI MARKET COMMITTEE in a fixed order (Skeptic last, after it can read
every other agent's findings) and the Decision Aggregator."""
from __future__ import annotations

from .base import AgentOutput
from .decision_agent.agent import aggregate
from .macro_agent.agent import MacroAgent
from .ml_agent.agent import MLScientistAgent
from .news_agent.agent import NewsAgent
from .portfolio_agent.agent import PortfolioAgent
from .quant_agent.agent import QuantAgent
from .risk_agent.agent import RiskAgent
from .skeptic_agent.agent import SkepticAgent
from .technical_agent.agent import TechnicalAgent

AGENTS = [QuantAgent, MacroAgent, NewsAgent, TechnicalAgent, RiskAgent, MLScientistAgent, PortfolioAgent]


def run_committee(state: dict) -> tuple[dict[str, AgentOutput], dict]:
    outputs: dict[str, AgentOutput] = {}
    for cls in AGENTS:
        a = cls()
        try:
            outputs[a.name] = a.run(state)
        except Exception as exc:  # noqa: BLE001 - an agent failure is reported, never hidden
            outputs[a.name] = AgentOutput(a.name, a.role, "NEUTRAL", 0.0, [], [], [f"agent failed: {exc}"])
    state = dict(state)
    state["agents"] = {k: v.to_dict() for k, v in outputs.items()}
    sk = SkepticAgent()
    outputs[sk.name] = sk.run(state)
    return outputs, aggregate(outputs, state.get("decision", {}))
