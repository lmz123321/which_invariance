def merge(records):
    record = records[0]
    for rec in records[1:]:
        for key in rec.keys():
            for subkey in rec[key].keys():
                record[key][subkey] += rec[key][subkey]
    return record