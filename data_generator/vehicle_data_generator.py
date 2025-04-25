import random
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker
import argparse

# Initialize Faker for Colombian locale
fake = Faker('es_CO')

# =========================
# Configuration Constants
# =========================

BRANDS = [
    'Chevrolet', 'Renault', 'Nissan', 'KIA',
    'Ford', 'Volkswagen', 'Hyundai', 'Toyota', 'Mazda'
]
BRANDS_WEIGHTS = [0.20, 0.18, 0.17, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04]

MODELS = {
    'Chevrolet':    ['Aveo', 'Spark', 'Sail', 'Tracker', 'Colorado', 'Onix'],
    'Renault':      ['Logan', 'Sandero', 'Duster', 'Kangoo', 'Stepway', 'Kwid'],
    'Nissan':       ['Sentra', 'Versa', 'Frontier', 'X-Trail', 'Kicks'],
    'KIA':          ['Rio', 'Cerato', 'Sportage', 'Picanto', 'Soul', 'Seltos'],
    'Ford':         ['Fiesta', 'Focus', 'Explorer', 'EcoSport', 'Ranger'],
    'Volkswagen':   ['Gol', 'Passat', 'Tiguan', 'Amarok', 'Polo'],
    'Hyundai':      ['i10', 'Elantra', 'Tucson', 'Accent', 'Kona'],
    'Toyota':       ['Corolla', 'Yaris', 'Hilux', 'Prado', 'Avanza'],
    'Mazda':        ['Mazda3', 'CX-3', 'CX-5', 'Mazda2', 'CX-30', 'CX-9']
}
MODELS_WEIGHTS = {
    'Chevrolet':    [0.20, 0.15, 0.15, 0.20, 0.15, 0.15],
    'Renault':      [0.20, 0.20, 0.25, 0.10, 0.10, 0.15],
    'Nissan':       [0.25, 0.25, 0.20, 0.15, 0.15],
    'KIA':          [0.20, 0.15, 0.20, 0.20, 0.15, 0.10],
    'Ford':         [0.20, 0.15, 0.20, 0.20, 0.25],
    'Volkswagen':   [0.25, 0.15, 0.20, 0.20, 0.20],
    'Hyundai':      [0.20, 0.20, 0.20, 0.20, 0.20],
    'Toyota':       [0.20, 0.15, 0.25, 0.20, 0.20],
    'Mazda':        [0.25, 0.25, 0.25, 0.10, 0.15, 0.00]
}

VEHICLE_TYPES = [
    'particular', 'servicio_publico', 'diplomatico',
    'remolque', 'carga_especial', 'suv', 'camioneta'
]

DEPARTMENTS_MUNICIPALITIES = {
    'Cundinamarca':    ['Bogotá'],
    'Antioquia':       ['Medellín', 'Bello', 'Envigado', 'Itagüí', 'Sabaneta'],
    'Valle del Cauca': ['Cali', 'Palmira', 'Yumbo', 'Jamundí'],
    'Atlántico':       ['Barranquilla', 'Soledad', 'Malambo', 'Puerto Colombia'],
    'Santander':       ['Bucaramanga', 'Floridablanca', 'Girón', 'Piedecuesta']
}

BODY_TYPES = ['Sedán', 'Hatchback', 'SUV', 'Camioneta', 'Bus']
BODY_TYPE_WEIGHTS = [0.30, 0.25, 0.35, 0.08, 0.02]

DEPRECIATION_RATES = {
    'Chevrolet': 0.08, 'Renault': 0.09, 'Toyota': 0.06,
    'Nissan': 0.085, 'KIA': 0.075, 'Ford': 0.095,
    'Volkswagen': 0.08, 'Hyundai': 0.07, 'Mazda': 0.07
}

# =========================
# Helper Functions
# =========================

def generate_plate(vehicle_type: str) -> str:
    if vehicle_type in ['particular', 'servicio_publico']:
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
        numbers = ''.join(random.choices('0123456789', k=3))
        return f"{letters} {numbers}"
    if vehicle_type == 'diplomatico':
        code = random.choice(['D','C','M','O','A'])
        letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        numbers = ''.join(random.choices('0123456789', k=4))
        return f"{code}{letter} {numbers}"
    if vehicle_type == 'remolque':
        code = random.choice(['R','S'])
        numbers = ''.join(random.choices('0123456789', k=5))
        return f"{code} {numbers}"
    if vehicle_type == 'carga_especial':
        numbers = ''.join(random.choices('0123456789', k=4))
        return f"T {numbers}"
    letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
    numbers = ''.join(random.choices('0123456789', k=3))
    return f"{letters} {numbers}"

def generate_color(brand: str) -> str:
    if brand in ['Chevrolet','Renault','Nissan','KIA','Ford','Volkswagen','Hyundai']:
        palette = ['Gris','Plata','Blanco','Rojo','Negro','Azul']
        weights = [0.35,0.35,0.10,0.10,0.05,0.05]
    elif brand == 'Toyota':
        palette = ['Gris','Plata','Blanco','Rojo','Negro','Azul']
        weights = [0.30,0.10,0.30,0.10,0.10,0.10]
    elif brand == 'Mazda':
        palette = ['Rojo','Gris','Plata','Blanco','Negro','Azul']
        weights = [0.70,0.10,0.05,0.05,0.05,0.05]
    else:
        palette = ['Gris/Plata','Blanco','Rojo','Otros']
        weights = [0.325,0.225,0.125,0.325]
    return random.choices(palette, weights=weights, k=1)[0]

def generate_vin() -> str:
    chars = 'ABCDEFGHJKLMNPRSTUVWXYZ1234567890'
    return ''.join(random.choices(chars, k=17))

def generate_chassis_number() -> str:
    return uuid.uuid4().hex.upper()[:12]

def generate_engine_number() -> str:
    return uuid.uuid4().hex.upper()[:12]

def generate_emission_standard(year: int) -> str:
    if year >= 2014:
        return random.choice(['Euro 4','Euro 5','EPA Tier 2'])
    if year >= 2010:
        return 'Euro 3'
    return 'Sin norma'

def assign_safety_features(year: int, seats: int, vehicle_type: str):
    abs_system = year >= 2012
    airbags = random.choice([0,2,4,6]) if year >= 2000 else 0
    esp = year >= 2015 or (vehicle_type == 'servicio_publico' and year >= 2020)
    seatbelts = seats
    return abs_system, airbags, esp, seatbelts

def generate_soat() -> tuple:
    validity = fake.date_between(start_date='-1y', end_date='+1y')
    policy = fake.bothify('SOAT-#####')
    return policy, validity

def generate_tech_inspection(vehicle_type: str):
    today = datetime.now().date()
    validity = today + timedelta(days=365 if vehicle_type=='servicio_publico' else 730)
    date = fake.date_between(start_date='-2y', end_date='today')
    result = random.choice(['Aprobado','Reprobado'])
    fur = fake.bothify('FUR-######')
    return date, result, validity, fur

def calculate_depreciation(year: int, brand: str, mileage: int) -> int:
    base = 50_000_000
    age = datetime.now().year - year
    rate = DEPRECIATION_RATES.get(brand, 0.08)
    value = base * ((1-rate)**age)
    factor = max(0.3, 1 - (mileage / 150_000))
    value *= factor
    return max(5_000_000, round(np.random.normal(loc=value, scale=value*0.10), -3))

def calculate_mileage(year: int, vehicle_type: str) -> int:
    age = datetime.now().year - year
    if vehicle_type == 'servicio_publico':
        mean, sd = age*25000, age*25000*0.3
    else:
        mean, sd = age*15000, age*15000*0.2
    return max(0, int(np.random.normal(mean, sd)))

def assign_seats_and_wheels(vehicle_type: str, sub_type: str=None):
    if vehicle_type == 'servicio_publico':
        seats = 4 if sub_type=='taxi' else random.choice([9,12,15,20,30])
        return seats, 4
    if vehicle_type in ['particular','diplomatico','suv','camioneta']:
        return random.choice([4,5,7]), 4
    if vehicle_type in ['remolque','carga_especial']:
        return random.choice([1,2]), random.choice([6,8,10,12])
    return 4,4

def assign_number_of_doors(vehicle_type: str, sub_type: str=None) -> int:
    if vehicle_type == 'servicio_publico':
        if sub_type == 'taxi':
            return 4
        if sub_type in ['bus','micro']:
            return random.choice([3,4,5])
    if vehicle_type in ['remolque','carga_especial']:
        return 2
    return 4

def assign_weight(vehicle_type: str) -> int:
    if vehicle_type == 'particular':
        return random.randint(1200, 1800)
    if vehicle_type in ['suv','camioneta']:
        return random.randint(1800, 2700)
    if vehicle_type in ['remolque','carga_especial']:
        return random.randint(1500, 3000)
    return random.randint(1200, 1800)

def assign_engine_displacement(vehicle_type: str) -> float:
    if vehicle_type == 'servicio_publico':
        return round(np.random.normal(1.6, 0.2), 1)
    if vehicle_type == 'suv':
        return round(np.random.normal(2.5, 0.3), 1)
    if vehicle_type == 'camioneta':
        return round(np.random.normal(2.2, 0.3), 1)
    if vehicle_type == 'carga_especial':
        return round(np.random.normal(3.0, 0.5), 1)
    return round(np.random.normal(1.5, 0.3), 1)

def assign_fuel_type(vehicle_type: str) -> str:
    if vehicle_type == 'servicio_publico':
        return random.choices(['Gasolina','Diesel','GNV'], weights=[0.2,0.3,0.5], k=1)[0]
    if vehicle_type in ['suv','camioneta']:
        return random.choices(['Gasolina','Diesel','Híbrido'], weights=[0.5,0.4,0.1], k=1)[0]
    if vehicle_type in ['remolque','carga_especial']:
        return 'Diesel'
    return random.choices(['Gasolina','Diesel','Eléctrico','Híbrido'], weights=[0.6,0.2,0.1,0.1], k=1)[0]

# =========================
# Vehicle Generation
# =========================

def generate_vehicle(sub_type: str=None) -> dict:
    brand = random.choices(BRANDS, weights=BRANDS_WEIGHTS, k=1)[0]
    model = random.choices(MODELS[brand], weights=MODELS_WEIGHTS.get(brand, [1]*len(MODELS[brand])), k=1)[0]
    year = random.randint(2000, datetime.now().year)
    vehicle_type = sub_type or random.choice(VEHICLE_TYPES)
    body = random.choices(BODY_TYPES, weights=BODY_TYPE_WEIGHTS, k=1)[0]
    dept = random.choice(list(DEPARTMENTS_MUNICIPALITIES.keys()))
    muni = random.choice(DEPARTMENTS_MUNICIPALITIES[dept])
    seats, wheels = assign_seats_and_wheels(vehicle_type, sub_type)
    plate = generate_plate(vehicle_type)
    color = generate_color(brand)
    vin = generate_vin()
    chassis = generate_chassis_number()
    engine_num = generate_engine_number()
    emission = generate_emission_standard(year)
    abs_sys, airbags, esp, belts = assign_safety_features(year, seats, vehicle_type)
    soat_pol, soat_vig = generate_soat()
    tech_date, tech_res, tech_vig, tech_fur = generate_tech_inspection(vehicle_type)
    mileage = calculate_mileage(year, vehicle_type)
    price = calculate_depreciation(year, brand, mileage)
    doors = assign_number_of_doors(vehicle_type, sub_type)
    weight = assign_weight(vehicle_type)
    engine_disp = assign_engine_displacement(vehicle_type)
    fuel = assign_fuel_type(vehicle_type)

    return {
        'vehicle_id':         str(uuid.uuid4()),
        'vin':                vin,
        'chassis_number':     chassis,
        'engine_number':      engine_num,
        'license_plate':      plate,
        'department':         dept,
        'municipality':       muni,
        'brand':              brand,
        'model':              model,
        'year':               year,
        'color':              color,
        'body_type':          body,
        'vehicle_type':       vehicle_type,
        'fuel_type':          fuel,
        'engine_displacement':engine_disp,
        'doors':              doors,
        'seats':              seats,
        'wheels':             wheels,
        'weight_kg':          weight,
        'mileage_km':         mileage,
        'price_cop':          price,
        'emission_standard':  emission,
        'abs':                abs_sys,
        'airbags':            airbags,
        'esp':                esp,
        'seatbelts':          belts,
        'soat_policy':        soat_pol,
        'soat_validity':      soat_vig,
        'techinsp_date':      tech_date,
        'techinsp_result':    tech_res,
        'techinsp_validity':  tech_vig,
        'fur_code':           tech_fur,
        'registration_date':  fake.date_between(start_date='-10y', end_date='today'),
        'status':             random.choice(['Nuevo','Usado'])
    }

def generate_dataset(n: int=1000, sub_type: str=None) -> pd.DataFrame:
    return pd.DataFrame([generate_vehicle(sub_type) for _ in range(n)])

# =========================
# Command–Line Interface
# =========================

def main():
    parser = argparse.ArgumentParser(description='Synthetic vehicle dataset generator for Colombia')
    parser.add_argument('-n', '--rows',   type=int,   default=1000, help='Number of records to generate')
    parser.add_argument('-o', '--output', type=str,   default='vehicle_data.csv', help='Output CSV file')
    parser.add_argument('-t', '--type',   type=str,   default=None, choices=VEHICLE_TYPES, help='Specific vehicle type to generate')
    args = parser.parse_args()

    df = generate_dataset(n=args.rows, sub_type=args.type)
    df.to_csv(args.output, index=False)
    print(f'Generated {args.rows} records in {args.output}')

if __name__ == '__main__':
    main()
