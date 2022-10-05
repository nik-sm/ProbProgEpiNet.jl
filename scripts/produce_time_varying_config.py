s1 = """
  "middlesex-exp-{}_cbgs": {{
    "county_pop": 1611699,
    "node_attributes": "size_varying/middlesex/{}_cbgs/01/node_attributes.json",
    "edge_attributes": "size_varying/middlesex/{}_cbgs/01/edge_attributes.json",
    "jhu_county_name": "Middlesex",
    "jhu_state_name": "Massachusetts",
    "mortality_rate": 0.0698,
    "death_undercount_factor": 10
  }},
  """

s2 = """
  "losangeles-exp-{}_cbgs": {{
    "county_pop": 10039107,
    "node_attributes": "size_varying/losangeles/{}_cbgs/01/node_attributes.json",
    "edge_attributes": "size_varying/losangeles/{}_cbgs/01/edge_attributes.json",
    "jhu_county_name": "Los Angeles",
    "jhu_state_name": "California",
    "mortality_rate": 0.0231,
    "death_undercount_factor": 10
  }},
  """
s3 = """
  "miamidade-exp-{}_cbgs": {{
    "county_pop": 2716940,
    "node_attributes": "size_varying/miamidade/{}_cbgs/01/node_attributes.json",
    "edge_attributes": "size_varying/miamidade/{}_cbgs/01/edge_attributes.json",
    "jhu_county_name": "Miami-Dade",
    "jhu_state_name": "Florida",
    "mortality_rate": 0.0157,
    "death_undercount_factor": 10
  }},
  """

with open("scrap_output.txt", "w") as f:
    for county in [s1, s2, s3]:
        for N in ["5", "10", "15", "30", "40"]:
            print(county.format(N, N, N))
            print(county.format(N, N, N), file=f)
