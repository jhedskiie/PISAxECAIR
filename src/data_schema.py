SCHEMA_GROUPS = [
    # Ordinal 1–6
    {
        "cols": ["SC001Q01TA"],
        "type": "ordinal",
        "min": 1,
        "max": 6,
        "post": ["round", "clip"],
    },

    # Binary coded as 1–2
    {
        "cols": [
            "SC013Q01TA",
            "SC053Q01TA", "SC053Q02TA", "SC053Q03TA", "SC053Q04TA", "SC053Q09TA", "SC053Q10TA",
        ],
        "type": "binary",
        "min": 1,
        "max": 2,
        "post": ["round", "clip"],
    },

    # Ordinal 1–4
    {
        "cols": [
            "SC017Q01NA", "SC017Q02NA", "SC017Q03NA", "SC017Q04NA",
            "SC017Q05NA", "SC017Q06NA", "SC017Q07NA", "SC017Q08NA",
            "SC155Q06HA", "SC155Q07HA", "SC155Q08HA", "SC155Q09HA",
            "SC155Q10HA", "SC155Q11HA",
            "SC061Q01TA", "SC061Q02TA", "SC061Q03TA", "SC061Q04TA",
            "SC061Q05TA", "SC061Q06TA", "SC061Q07TA", "SC061Q08TA",
            "SC061Q09TA", "SC061Q10TA", "SC061Q11HA",
        ],
        "type": "ordinal",
        "min": 1,
        "max": 4,
        "post": ["round", "clip"],
    },

    # Ordinal 1–3
    {
        "cols": [
            "SC011Q01TA",
            "SC012Q01TA", "SC012Q02TA", "SC012Q03TA", "SC012Q04TA", "SC012Q05TA", "SC012Q06TA",
            "SC042Q01TA", "SC042Q02TA",
            "SC037Q01TA", "SC037Q02TA", "SC037Q03TA", "SC037Q04TA",
            "SC037Q05NA", "SC037Q06NA", "SC037Q07TA", "SC037Q08TA", "SC037Q09TA",
        ],
        "type": "ordinal",
        "min": 1,
        "max": 3,
        "post": ["round", "clip"],
    },

    # Count 0–10000
    {
        "cols": [
            "SC002Q01TA", "SC002Q02TA",
            "SC004Q01TA", "SC004Q02TA", "SC004Q03TA", "SC004Q05NA", "SC004Q06NA", "SC004Q07NA",
            "SC018Q01TA01", "SC018Q01TA02", "SC018Q02TA01", "SC018Q02TA02",
        ],
        "type": "count",
        "min": 0,
        "max": 10000,
        "post": ["round", "clip"],
    },

    # Count 0–100
    {
        "cols": [
            "SC025Q01NA", "SC064Q01TA", "SC064Q02TA", "SC064Q03TA", "SC064Q04NA",
        ],
        "type": "count",
        "min": 0,
        "max": 100,
        "post": ["round", "clip"],
    },

    # Ordinal 1–9
    {
        "cols": ["SC003Q01TA"],
        "type": "ordinal",
        "min": 1,
        "max": 9,
        "post": ["round", "clip"],
    },
]

COMPOSITION_GROUPS = {
    "SC016": {
        "cols": ["SC016Q01TA", "SC016Q02TA", "SC016Q03TA", "SC016Q04TA"],
        "sum": 100,
    }
}