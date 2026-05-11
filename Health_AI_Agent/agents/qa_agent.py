def validate_sql(sql):

    forbidden=["DELETE","DROP","UPDATE"]

    for f in forbidden:
        if f in sql.upper():
            raise Exception("Unsafe SQL")

    return True
