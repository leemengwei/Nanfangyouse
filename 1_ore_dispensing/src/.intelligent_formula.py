from IPython import embed
import argparse
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from sys import stderr
import build_model
import pandas as pd
from multiprocessing import cpu_count
import torch
import sys

app = Flask(__name__)

@app.route('/api/calculate', methods=['POST', 'GET'])
@cross_origin()
def calculate():
    req_data = request.get_json()
    print(req_data, file=stderr)
 
    #req_data to pandas:
    INGREDIENT_STORAGE = pd.DataFrame() 
    for i in req_data['list']: 
        INGREDIENT_STORAGE = INGREDIENT_STORAGE.append(pd.DataFrame(data=i, index=[i['name']]))

    #Call GA:
    #best_ratio, best_solution = build_model.run_opt(args)
    
    #res_data to pandas

    res_data = {
        "list": 
        [
            {
                "name": "水星轮",
                "number": 10001,
                "inventoryBalance": 1346,
                "calculatePercentage": 31.23
            },
            {
                "name": "莱科塔",
                "number": 10002,
                "inventoryBalance": 5686,
                "calculatePercentage": 51.23
            }
        ],
        "calculateParameter":
        {
            "oxygenMaterialRatio": 12.32,
        },
        "elementsMixtureList": 
        [
            {
                "name": "Cu",
                "percentage": 24.01
            },
            {
                "name": "Fe",
                "percentage": 31.02
            }
        ]
    }
    return jsonify(res_data)

@app.route('/api/getInventory', methods=['GET'])
@cross_origin()
def getInventory():
    #获取库存 for 显示
    INVENTORY_STORAGE = build_model.get_storage(for_show=True)

    #pandas to res_data
    res_data = [] 
    key_altering = {'ratio':'calculatePercentage'}
    for i in INVENTORY_STORAGE.iterrows(): 
        this_dict = {}
        this_dict['name'] = i[0]
        for this_attr in i[1].index: 
            key_attr = this_attr
            if key_attr in key_altering.keys():
                key_attr = key_altering[key_attr]
            this_dict[key_attr] = i[1][this_attr] 
        res_data.append(this_dict) 

   # Sample:
   # res_data = [
   #     {
   #         "required": True,
   #         "name": "水星轮",
   #         "number": 10001,
   #         "Cu": 1,
   #         "Fe": 2,
   #         "S": 3,
   #         "SiO2": 4,
   #         "Cao": 5,
   #         "As": 6,
   #         "Zn": 7,
   #         "Pb": 8,
   #         "MgO": 9,
   #         "Al2O3": 10,
   #         "H2O": 11,
   #         "inventory": 12,
   #         "calculatePercentage": 5
   #     },
   # ]
    return jsonify(res_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", '--DEBUG', action='store_true', default=False)
    parser.add_argument("-O", '--OBJ', type=int, default=0)
    parser.add_argument("-E", '--epoch', type=int, default=50)
    parser.add_argument("-P", '--pop', type=int, default=100)
    parser.add_argument("-A", '--alpha', type=int, default=1)
    parser.add_argument("-B", '--beta', type=int, default=1)
    parser.add_argument("-G", '--gama', type=int, default=1)
    parser.add_argument("-T", '--threads', type=int, default=int(cpu_count()/2))
    parser.add_argument("-M", '--MAX_TYPE_ALLOWED', type=int, default=4)
    parser.add_argument("-ELEMENTS", '--ELEMENTS', type=list, default=['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O'])
    parser.add_argument("-ELEMENTS_MATTERS", '--ELEMENTS_MATTERS', type=list, default=['Cu', 'Fe', 'S', 'SiO2', 'CaO', 'As', 'Zn', 'Pb', 'MgO', 'Al2O3', 'H2O'])
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #获取库存 for 计算
    args.INGREDIENT_STORAGE = build_model.get_storage()
    args.INGREDIENT_CHOOSE_FROM = list(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.ratio==0].index)
    args.INGREDIENT_MUST = list(args.INGREDIENT_STORAGE[args.INGREDIENT_STORAGE.ratio!=0].index)
    args.NUM_OF_TYPES_FOR_GA = len(args.INGREDIENT_CHOOSE_FROM)
    #获取元素配比目标
    args.ELEMENT_TARGETS_LOW, args.ELEMENT_TARGETS_MEAN, args.ELEMENT_TARGETS_HIGH = build_model.get_ELEMENT_TARGETS(args)

    app.run(host='0.0.0.0', port=7001, debug=True)



