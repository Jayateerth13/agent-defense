"""Manual helper: print the transcript + Progent events of one logged pair. Usage: python tests/_show_run.py user_task_13 injection_task_5"""
import json
import sys

u, i = sys.argv[1], sys.argv[2]
rs = [json.loads(l) for l in open('results/raw_logs.jsonl', encoding='utf-8') if l.strip()]
for r in rs:
    if r['user_task'] == u and r['injection_task'] == i:
        print('=====', r['condition'], 'utility', r['utility'], 'attack', r['attack_success'])
        for m in r['messages']:
            role = m.get('role')
            c = m.get('content') or ''
            tc = m.get('tool_calls')
            if role == 'system':
                continue
            print(f'[{role}]', str(c)[:700].replace('\n', ' | '), json.dumps(tc)[:300] if tc else '')
        if r.get('progent'):
            p = r['progent']
            print('INITIAL', json.dumps(p['initial_policy'])[:900])
            for e in p['events']:
                print('EV', json.dumps(e)[:600])
