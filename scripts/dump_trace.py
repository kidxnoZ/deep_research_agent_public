import json, sys
path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    msgs = json.load(f)
for i, msg in enumerate(msgs):
    role = msg['role']
    content = msg['content']
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get('type')
            if t == 'tool_use':
                inp = block.get('input', {})
                brief = {k: (str(v)[:50] if isinstance(v, str) else v)
                         for k, v in inp.items() if k not in ('sections', 'draft_content', 'critique_feedback')}
                name = block.get('name', '')
                print(f"[msg {i}] {role} -> {name}({json.dumps(brief, ensure_ascii=False)})")
            elif t == 'tool_result':
                try:
                    r = json.loads(block['content'])
                    if 'is_sufficient' in r:
                        print(f"  <- critique: sufficient={r['is_sufficient']} score={r.get('quality_score')} iter={r.get('iteration')}")
                        for g in r.get('gaps', [])[:2]:
                            print(f"     gap: {g[:80]}")
                    elif 'saved' in r:
                        print(f"  <- save_section: saved={r['saved']} done={r.get('sections_completed')}/{r.get('sections_total')}")
                    elif 'source_id' in r:
                        print(f"  <- search: {r.get('source_id')} summary={str(r.get('summary',''))[:50]}")
                    elif 'initialized' in r:
                        print(f"  <- criteria: initialized={r['initialized']}")
                    elif 'sections' in r or 'plan' in str(r.get('note','')):
                        print(f"  <- plan_sections result")
                    elif 'error' in r:
                        print(f"  <- ERROR: {str(r['error'])[:100]}")
                    else:
                        print(f"  <- {list(r.keys())[:5]}")
                except Exception as e:
                    print(f"  <- (parse err: {e})")
            elif t == 'text' and block.get('text', '').strip():
                print(f"[msg {i}] {role} [text] {block['text'][:100]}")
