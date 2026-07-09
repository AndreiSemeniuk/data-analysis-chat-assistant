Your previous BigQuery SQL attempt failed. Fix it.

## Database schema
{schema}

## User question
{question}

## Previous SQL
```sql
{previous_sql}
```

## What went wrong
{error}

Common causes: wrong column name (check the schema above), BigQuery dialect issues, an impossible join, or a filter that matches nothing (if the result was empty, relax the most restrictive filter or check spelling of literal values).

Respond with the corrected SQL only, inside a ```sql code block.
