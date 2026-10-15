"""Shared context path and hard tool gates for reproducible FAB/LEC ablations."""
from __future__ import annotations
import os,re
from typing import Any
from .memory_store import build_inject_summary, is_first_user_message_in_session, memory_inject_enabled
from .skill_store import build_skill_inject, skill_inject_enabled
from .lec_store import build_evidence_context, lec_enabled
from .reliability import request_text

def fab_enabled() -> bool:
    return os.getenv("POLYSEA_FAB_ENABLED","1").lower() not in {"0","false","off","no"}

def tool_allowed(name: str) -> bool:
    if name.startswith(("memory_","skill_")) and not fab_enabled():return False
    if name.startswith(("deepxiv_","lec_")) and not lec_enabled():return False
    return True

def enabled_tools(tools: list[dict[str,Any]]) -> list[dict[str,Any]]:
    return [t for t in tools if tool_allowed(t.get('function',{}).get('name',''))]

def assemble_context(messages: list[dict[str,Any]]) -> tuple[str,dict[str,Any]]:
    chunks=[];trace={"fab_enabled":fab_enabled(),"lec_enabled":lec_enabled(),"memory_chars":0,"skill_chars":0,"evidence_ids":[]}
    if fab_enabled():
        if memory_inject_enabled() and is_first_user_message_in_session(messages):
            value=build_inject_summary()
            if value:chunks.append(value);trace['memory_chars']=len(value)
        if skill_inject_enabled():
            value=build_skill_inject(messages)
            if value:chunks.append(value);trace['skill_chars']=len(value)
    else:chunks.append('[Persistent FAB memory and skill tools are disabled for this run.]')
    query=request_text(messages)
    if lec_enabled() and re.search(r"literature|paper|source|citation|cite|evidence|文献|论文|来源|依据|引用",query,re.I):
        value,ids=build_evidence_context(query)
        if value:chunks.append(value);trace['evidence_ids']=ids
    elif not lec_enabled():chunks.append('[LEC is disabled for this run. Do not claim that literature tools were queried.]')
    return '\n\n'.join(chunks),trace
