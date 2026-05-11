def analyze(data):

    if not data:
        return "No records"

    return f"{len(data)} rows returned"
