from threading import Lock

counter_lock = Lock()
zone_counters = {
    "zone1": {"entrance": 0, "exit": 0, "zone_count": 0},
    "zone2": {"entrance": 0, "exit": 0, "zone_count": 0}
}