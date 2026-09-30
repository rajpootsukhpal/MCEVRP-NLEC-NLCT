from __future__ import annotations
from dataclasses import dataclass

# Case definitions copied from ../real_case_noida.py; no network or output side effects.
@dataclass
class Location:
    node: str
    kind: str
    name: str
    lat: float
    lon: float
    address: str
    source: str

def noida_real_locations() -> list[Location]:
    return [
        Location(
            node="Depot",
            kind="depot",
            name="Sector 88 Fruit & Vegetable Mandi Cluster",
            lat=28.531111,
            lon=77.425000,
            address="Sector 88, near fruit & vegetable mandi, Noida, Uttar Pradesh",
            source="Sector 88 locality/mandi cluster references",
        ),
        Location(
            node="C1",
            kind="customer",
            name="Marks & Spencer, Sector 18",
            lat=28.567970,
            lon=77.320513,
            address="Chilla, Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C2",
            kind="customer",
            name="Shopprix Mall",
            lat=28.597126,
            lon=77.364919,
            address="Maharaja Agrasen Marg, Mamura, Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C3",
            kind="customer",
            name="Organic India Noida",
            lat=28.541016,
            lon=77.366989,
            address="Vishwakarma Road, Hazipur, Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C4",
            kind="customer",
            name="Commercial 1, Paras Tierea",
            lat=28.507493,
            lon=77.413224,
            address="Chandila Marg, Shahdara, Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C5",
            kind="customer",
            name="Galaxy Shoppe",
            lat=28.619571,
            lon=77.425960,
            address="Mahagun Mart service cluster, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C6",
            kind="customer",
            name="Stellar Jeevan Retail Cluster",
            lat=28.565432,
            lon=77.447957,
            address="Panchmukhi Chowk, Greater Noida West, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C7",
            kind="customer",
            name="Habitech Qube Crystal Mall",
            lat=28.481902,
            lon=77.479746,
            address="Knowledge Park III, Greater Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C8",
            kind="customer",
            name="Phase 1 Market",
            lat=28.444405,
            lon=77.511348,
            address="Yamuna Expressway, Greater Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C9",
            kind="customer",
            name="Supertech Mart",
            lat=28.610201,
            lon=77.441602,
            address="Service Road, Greater Noida West, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C10",
            kind="customer",
            name="PK Residency",
            lat=28.536994,
            lon=77.365943,
            address="Hazipur, Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C11",
            kind="customer",
            name="Delhi ENT Multispeciality Hospital",
            lat=28.539926,
            lon=77.292145,
            address="District Centre, Jasola fringe, South East Delhi",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C12",
            kind="customer",
            name="Avantika Hospital",
            lat=28.645553,
            lon=77.371756,
            address="Vasundhara, Ghaziabad, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C13",
            kind="customer",
            name="ANJULI Nursing Home",
            lat=28.400031,
            lon=77.368855,
            address="Sector 78 fringe, Faridabad, Haryana",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C14",
            kind="customer",
            name="Pacific Mall",
            lat=28.646306,
            lon=77.320255,
            address="Vivek Vihar border, Delhi NCR",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C15",
            kind="customer",
            name="Modi Mall",
            lat=28.586298,
            lon=77.341161,
            address="Sector 10, Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C16",
            kind="customer",
            name="Metro Multi Speciality Hospital",
            lat=28.406521,
            lon=77.317923,
            address="Sector 16A fringe, Faridabad, Haryana",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C17",
            kind="customer",
            name="Ashira Inn",
            lat=28.476699,
            lon=77.476817,
            address="Knowledge Park III, Greater Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C18",
            kind="customer",
            name="Sandal Hotel",
            lat=28.495560,
            lon=77.402621,
            address="Bajidpur, Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C19",
            kind="customer",
            name="Radisson Blu Greater Noida",
            lat=28.449850,
            lon=77.527765,
            address="Sector 20, Greater Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C20",
            kind="customer",
            name="Mayur Medicare Center",
            lat=28.606058,
            lon=77.293380,
            address="Mayur Vihar fringe, East Delhi",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C21",
            kind="customer",
            name="L.D. Hospital",
            lat=28.500402,
            lon=77.323280,
            address="Badarpur fringe, South East Delhi",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C22",
            kind="customer",
            name="Mohan Swarup Hospital",
            lat=28.566876,
            lon=77.541120,
            address="Dadri Bypass, Dadri, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C23",
            kind="customer",
            name="Bennett University",
            lat=28.450643,
            lon=77.583798,
            address="Madhaiya Maincha, Dadri, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C24",
            kind="customer",
            name="Shiv Nadar University",
            lat=28.525806,
            lon=77.574753,
            address="Chithera, Dadri, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C25",
            kind="customer",
            name="Shikha Acupuncture Sanatorium",
            lat=28.653957,
            lon=77.471763,
            address="NH9 site cluster, Ghaziabad border",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C26",
            kind="customer",
            name="Government Hospital Dujana",
            lat=28.622719,
            lon=77.509977,
            address="Grand Trunk Road, Dadri, Gautam Buddha Nagar, Uttar Pradesh",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="C27",
            kind="customer",
            name="Sakshi Medicare",
            lat=28.686392,
            lon=77.416909,
            address="Nandagram Road, Ghaziabad border",
            source="OpenStreetMap / Nominatim",
        ),
        Location(
            node="S1",
            kind="station",
            name="Tata Power EV Charging Station, Wave One",
            lat=28.570486,
            lon=77.323653,
            address="Wave One, Plot L2A, Sector 18, Noida, Uttar Pradesh",
            source="Mappls EV charging listing",
        ),
        Location(
            node="S2",
            kind="station",
            name="NTPC EV Charging Station, Ecotech II",
            lat=28.504521,
            lon=77.464185,
            address="E3, Ecotech II, Udyog Vihar, Greater Noida, Uttar Pradesh",
            source="Mappls EV charging listing",
        ),
        Location(
            node="S3",
            kind="station",
            name="Adani Gas EV Charging Station, Gaur City Mall",
            lat=28.605664,
            lon=77.429139,
            address="Gaur City Mall Multi Level Car Parking, Sector 04, Greater Noida West, Uttar Pradesh",
            source="Mappls EV charging listing",
        ),
        Location(
            node="S4",
            kind="station",
            name="Tata Power EV Charging Station, Pari Chowk",
            lat=28.468119,
            lon=77.506166,
            address="Pari Chowk, Greater Noida, Gautam Buddha Nagar, Uttar Pradesh",
            source="Public charging map listing",
        ),
    ]

def build_real_case_data(
    battery_capacity_value: float = 14.4,
    charge_time_weight: float = 1.0,
    demand_scale: float = 1.0,
    fleet_size: int = 7,
    distance_duration: tuple[dict, dict] | None = None,
):
    locations = noida_real_locations()
    if distance_duration is None:
        raise ValueError("Provide the archived road matrices")
    distances, durations = distance_duration
    depot = "Depot"
    selected_customer_ids = [
        "C6",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
        "C22",
        "C23",
        "C24",
        "C25",
        "C26",
        "C27",
    ]
    customers = selected_customer_ids
    stations = [loc.node for loc in locations if loc.kind == "station"]
    products = ["A", "B"]
    compartments = ["L1", "L2"]
    vehicles = [f"V{i+1}" for i in range(fleet_size)]
    breakpoints = ["B0", "B1", "B2", "B3"]
    nodes = [loc.node for loc in locations]
    non_depot_nodes = [n for n in nodes if n != depot]

    def demand_profile(customer: str) -> dict[str, float]:
        loc = next(loc for loc in locations if loc.node == customer)
        name = loc.name.lower()
        if "university" in name:
            return {"A": 55.0, "B": 42.0}
        if "hospital" in name:
            return {"A": 48.0, "B": 36.0}
        if "hotel" in name or "inn" in name:
            return {"A": 44.0, "B": 31.0}
        if "mall" in name or "mart" in name or "market" in name or "shop" in name:
            return {"A": 52.0, "B": 39.0}
        return {"A": 40.0, "B": 30.0}

    demand = {depot: {"A": 0.0, "B": 0.0}}
    for customer in customers:
        base = demand_profile(customer)
        demand[customer] = {p: round(base[p] * demand_scale, 2) for p in products}
    for station in stations:
        demand[station] = {"A": 0.0, "B": 0.0}

    vehicle_capacity = {v: 750.0 for v in vehicles}
    compartment_capacity = {(v, "L1"): 375.0 for v in vehicles}
    compartment_capacity.update({(v, "L2"): 375.0 for v in vehicles})
    compatibility = {(l, p): 1 for l in compartments for p in products}
    battery_capacity = {v: battery_capacity_value for v in vehicles}
    safety_soc = {v: 1.8 for v in vehicles}
    fixed_cost = {v: 55.0 for v in vehicles}
    service_time = {node: 0.0 for node in nodes}
    for customer in customers:
        service_time[customer] = 0.5

    time_window = {node: (0.0, 24.0) for node in nodes}
    for customer in customers:
        time_window[customer] = (3.0, 17.5)

    route_cost = dict(distances)
    travel_time = dict(durations)
    alpha = {(i, j): 0.070 * distances[(i, j)] for (i, j) in distances}
    beta = {(i, j): 0.00011 * distances[(i, j)] for (i, j) in distances}
    charge_price = {"S1": 0.60, "S2": 0.55, "S3": 0.58, "S4": 0.59}
    tardiness_weight = 8.0

    frac_levels = {"B0": 0.0, "B1": 0.4, "B2": 0.8, "B3": 1.0}
    rho = {b: battery_capacity_value * frac_levels[b] for b in breakpoints}
    full_charge_hours = {"S1": 6.5, "S2": 6.2, "S3": 6.4, "S4": 6.3}
    gamma = {}
    for station in stations:
        for b in breakpoints:
            frac = frac_levels[b]
            gamma[(station, b)] = full_charge_hours[station] * (0.45 * frac + 0.55 * frac * frac)

    return {
        "case_name": "noida_greater_noida_bulk_grocery_corridor_real_case",
        "depot": depot,
        "customers": customers,
        "stations": stations,
        "vehicles": vehicles,
        "products": products,
        "compartments": compartments,
        "breakpoints": breakpoints,
        "nodes": nodes,
        "non_depot_nodes": non_depot_nodes,
        "distance": distances,
        "travel_time": travel_time,
        "route_cost": route_cost,
        "demand": demand,
        "vehicle_capacity": vehicle_capacity,
        "compartment_capacity": compartment_capacity,
        "battery_capacity": battery_capacity,
        "safety_soc": safety_soc,
        "service_time": service_time,
        "time_window": time_window,
        "fixed_cost": fixed_cost,
        "charge_price": charge_price,
        "charge_time_weight": charge_time_weight,
        "tardiness_weight": tardiness_weight,
        "alpha": alpha,
        "beta": beta,
        "compatibility": compatibility,
        "rho": rho,
        "gamma": gamma,
        "big_m": 1.0e5,
        "locations": {loc.node: loc for loc in locations},
        "customer_alias": {node: f"C{idx + 1}" for idx, node in enumerate(customers)},
    }