cd generate
# python generate_o3.py
python generate.py
cp output.json ../driver/output.json
cd ..
cd driver
python driver.py
