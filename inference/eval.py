def calculate_SLO_hit(latency_hits):

	return sum(latency_hits)/(len(latency_hits)*1.0)

