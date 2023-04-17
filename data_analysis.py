
result = {
        'Medical Record': 0,
        'Animal': 0,
        'Species': 0,
        'Breed': 0,
        'Disease': 0,
        'Symptom': 0,
        'Drugs': 0,
        'Prescription': 0,
        'Treatment code': 0,
        'Treatment': 0,
        'Comment': 0,
        'Age': 0,
        'Age Group': 0,
        'Gender': 0,
        'Weight': 0,
        'Disease Category': 0,
    }


with open('data/Balance_800/entities.txt', encoding='utf8') as f:
    while True:
        line = f.readline()

        data = line.split("_")
        type_ = data[0]

        if not type_:
            break
        if "diagnosis" == type_ and "cc" not in line:
            result['Medical Record'] += 1
        elif "pet" == type_:
            result['Animal'] += 1
        elif "s" == type_:
            
            result['Species'] += 1
        elif "b" == type_:
            
            result['Breed'] += 1
        elif "gender" == type_:
            
            result['Gender'] += 1
        elif "tx" == type_ and "code" not in line:
            result['Treatment'] += 1
        elif "tx" == type_:
           
            result['Treatment code'] += 1
        elif "type" == type_:
            result['Disease Category'] += 1
        elif "memo" == type_:
            result['Symptom'] += 1
        elif "ag" == type_ and "ag_ag_" not in line:
            result['Age Group'] += 1
        elif "age" == type_:
            result['Age'] += 1
        elif "diagnosis" == type_:
            result['Comment'] += 1
        elif "rx" == type_ and "code" not in line:
            result['Prescription'] += 1
        elif "rx" == type_:
            print(line)
            result['Drugs'] += 1
        elif "weight" != type_ and "b" != type_ and "s" != type_ and "ag" != type_:
            
            result['Disease'] += 1
        else :
            result['Weight'] += 1


print(result)