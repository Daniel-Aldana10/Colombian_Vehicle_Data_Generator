import unittest
import random
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime


from data_generator.vehicle_data_generator import (
    generate_vehicle,
    generate_dataset,
    assign_fuel_type,
    DEPARTMENTS_MUNICIPALITIES,
    generate_deterministic_plate,
    generate_deterministic_soat
)


class TestVehicleGenerator(unittest.TestCase):
    def setUp(self):
        random.seed(123)
        np.random.seed(123)
        self.sample = generate_vehicle(index=12345)

    def test_required_fields(self):
        expected = {
            'vehicle_id','vin','chassis_number','engine_number','license_plate',
            'department','municipality','brand','model','year','color','body_type',
            'vehicle_type','fuel_type','engine_displacement','doors','seats','wheels',
            'weight_kg','mileage_km','price_cop','emission_standard','abs','airbags',
            'esp','seatbelts','soat_policy','soat_validity','techinsp_date',
            'techinsp_result','techinsp_validity','fur_code','registration_date','status'
        }
        missing = expected - set(self.sample.keys())
        self.assertFalse(missing, f"Missing fields: {missing}")

    def test_year_range(self):
        year = self.sample['year']
        current = datetime.now().year
        self.assertTrue(2000 <= year <= current)

    def test_municipality_in_department(self):
        dept = self.sample['department']
        muni = self.sample['municipality']
        self.assertIn(muni, DEPARTMENTS_MUNICIPALITIES[dept])

    def test_doors_logic(self):
        vt = self.sample['vehicle_type']
        doors = self.sample['doors']
        if vt in ['remolque','carga_especial']:
            self.assertEqual(doors, 2)
        else:
            self.assertIn(doors, [2,3,4,5])

    def test_weight_ranges(self):
        wt = self.sample['weight_kg']
        vt = self.sample['vehicle_type']
        if vt == 'particular':
            self.assertTrue(1200 <= wt <= 1800)
        elif vt == 'suv':
            self.assertTrue(1800 <= wt <= 2700)

    def test_engine_displacement(self):
        disp = self.sample['engine_displacement']
        vt = self.sample['vehicle_type']
        if vt == 'remolque':
            self.assertEqual(disp, 0.0)
        else:
            self.assertGreater(disp, 0)

    def test_fuel_type_distribution(self):
        trials = 500
        fuels = [
            assign_fuel_type(random.choice(['particular','suv','servicio_publico']))
            for _ in range(trials)
        ]
        freq = pd.Series(fuels).value_counts(normalize=True)
        self.assertGreater(freq.get('Gasolina', 0), 0.4)
        self.assertGreater(freq.get('Diesel',   0), 0.2)

    def test_deterministic_plate_uniqueness(self):
        # Test that plates are unique for different indices
        plates = [generate_deterministic_plate(i, 'particular') for i in range(100)]
        self.assertEqual(len(plates), len(set(plates)), "Plates should be unique")
        
    def test_deterministic_soat_uniqueness(self):
        # Test that SOAT policies are unique for different indices
        soats = [generate_deterministic_soat(i)[0] for i in range(100)]
        self.assertEqual(len(soats), len(set(soats)), "SOAT policies should be unique")

    def test_dataset_generation_writes_to_file(self):
        n = 50
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            generate_dataset(n=n, output_path=temp_path)
            
            # Verify file exists and has correct number of rows
            self.assertTrue(os.path.exists(temp_path))
            df = pd.read_csv(temp_path)
            self.assertEqual(len(df), n)
            self.assertFalse(df.isnull().any().any())
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
