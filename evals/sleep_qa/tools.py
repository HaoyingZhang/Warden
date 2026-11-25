"""The agent's only tool. It runs `tsq` inside the sandbox's agent container, which forwards the
query to the gateway; nothing here talks to InfluxDB directly."""

from inspect_ai.tool import tool
from inspect_ai.util import sandbox


@tool
def query_influxdb():
    async def execute(flux: str, max_rows: int = 200) -> str:
        """Run a read-only Flux query on the user's sleep database and return the rows as JSON.

        Only bucket "eeg" can be read. Writing with to() and imports other than date, strings,
        math, regexp, timezone and influxdata/influxdb/schema are rejected. Results are cut at
        max_rows rows, so aggregate in Flux (pivot, aggregateWindow, mean, sum, count) instead of
        fetching raw data.

        Args:
            flux: The complete Flux query.
            max_rows: Maximum number of rows to return, at most 1000.
        """
        result = await sandbox().exec(["tsq", str(int(max_rows))], input=flux, timeout=180)
        return result.stdout if result.success else f"Error: {result.stderr.strip()}"

    return execute
