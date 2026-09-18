import params

def velocity_to_distance(velocity_mm_per_second):
    # Convert mms per second to feet per second.
    fps = velocity_mm_per_second / 304.8

    return params.STIMP * (fps ** 2) / (6.0 ** 2)
