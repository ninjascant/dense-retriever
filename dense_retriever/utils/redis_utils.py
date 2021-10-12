import redis
import json


class RedisClient:
    def __init__(self, hostname, port=6379, username=None, passwd=None):
        self.r = redis.Redis(
            host=hostname,
            port=port,
            username=username,
            password=passwd,
            # ssl=True,
            # ssl_cert_reqs=None
        )

    def read(self, key):
        return json.loads(self.r.get(key))

    def write(self, key, value):
        value_str = json.dumps(value)
        self.r.set(key, value_str)
