import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'texture_mean':10.38, 'area_mean':1001, 'concavity_mean':0.3001, 'area_se':153.4, 'smoothness_worst':0.1622,'concavity_worst':0.7119, 'symmetry_worst':0.4601,'radius_mean':9.787,'perimeter_mean':79.78,'concave points_mean':0.02941,'radius_worst': 15.110,'texture_worst':19.26, 'perimeter_worst':65.13, 'area_worst':314.9,'concave points_worst':0.06227})

print(r.json())

# ['!radius_mean', '!texture_mean', '!perimeter_mean', '!area_mean',
#      '!concavity_mean', '!concave points_mean', '!area_se', '!radius_worst',
#      'texture_worst', 'perimeter_worst', 'area_worst', '!smoothness_worst',
#      '!concavity_worst', 'concave points_worst', '!symmetry_worst']
