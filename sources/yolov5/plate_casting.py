def swic_recor(car):
    switcher = {
        # 'NG': "Xe ngoại giao",
        # 'NN': "Xe nước ngoài",
        # 'QT': "Xe quốc tế",
        # 'CV': "Xe công vụ",
        'NG': "3",
        'NN': "4",
        'QT': "7",
        'CV': "6",
    }
    return switcher.get(car)

def swic_color(vehicle):
    switcher = {
        # 'white': "Xe dân sự",
        # 'yellow': "Xe dịch vụ",
        # 'blue': "Xe công an, Nhà Nước",
        # 'red': "Xe quân sự",
        "white": "0",
        "blue": "2",
        "red": "1",
        "yellow": "5"
    }
    return switcher.get(vehicle)


def check_plate(plate, color, cls):
    ch = ''
    for c in plate:
        if 'A' <= c <= 'Z':
            ch += c
    if swic_recor(ch) is not None:
        veh = swic_recor(ch)
    else:
        if int(cls) == 1 and color == "yellow":
            veh = "0"
        else:
            veh = swic_color(color)
    return veh

def classes(cls):
    if int(cls) == 3:
        return 1
    else: 
        return 0
    