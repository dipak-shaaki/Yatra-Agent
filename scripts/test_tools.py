from app.agent.tools.check_scope_tool import check_scope
from app.agent.tools.search_destinations_tool import search_destinations

print(check_scope("What permits do I need for Manaslu?"))
print(check_scope("Can you book me a hotel in Pokhara?"))
print()
for r in search_destinations("What's the best time to visit Rara Lake?", n_results=2):
    print(r["destination"], "-", r["section"])
    print(r["text"][:120])
    print()
