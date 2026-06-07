import sys
try:
    import core.automl as a
    import core.goal_agent as g
    print('import_ok')
except Exception as e:
    print('ERR', type(e).__name__, e)
    sys.exit(1)
