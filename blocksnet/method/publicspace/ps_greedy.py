import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import shapely

from ...models import Block, ServiceType
from ..base_method import BaseMethod


class PublicSpaceGreedy:
    def __init__(self, vacant_area, buildings):
        # iso: изохроны в формате geojson с разным временем
        # OPM: файл с потенциальными общественными пространствами
        # house: дома

        self.iso_10 = None
        self.iso_20 = None
        self.iso_5 = None
        self.OPM = vacant_area
        for index, row in self.OPM.iterrows():
            self.OPM.loc[index, "functional_type"] = ""
            self.OPM.loc[index, "exists"] = "потенциальное"

        house = buildings
        house = house.to_crs(4326)
        house["house_id"] = house.index
        self.house = house

        self.is_graph_exist = False
        self.street_graph = None
        self.graph = None
        self.existed = []

    # Функция для автоматической классификации существующих ОПМ
    def find_exist(self, services_path):
        services = gpd.read_file(services_path)
        opm = self.OPM[self.OPM["Городская функция"] == "Природа"]
        opm = opm.to_crs(3857)
        buf = opm.buffer(200).to_crs(4326)
        buf = gpd.GeoDataFrame(buf)
        buf.columns = ["geometry"]
        buf.set_geometry("geometry", inplace=True)
        buf["id"] = opm["id"]
        a = gpd.sjoin(buf, services)

        dic = {}
        for i in a.id_left.unique():
            dic[int(i)] = []
        for id, row in a.iterrows():
            key = row["id_left"]
            dic[key].append(row["index_right"])

        M = ["Клуб для детей и подростков", "Торгово-развлекательный центр", "Парк развлечений", "Зоопарк"]
        U = ["Музыкальная школа", "Библиотека", "Музей"]
        S = [
            "Театр/Концертный зал",
            "Музей",
            "Кинотеатр",
            "Квест",
            "Картинная галерея",
            "Арт пространство",
            "Аквапарк",
            "Цирк",
            "Боулинг",
        ]
        Sp = ["Спортивная секция", "Спортивный центр", "Бассейн", "Стадион"]

        type_dic = {}
        for i in a.id_left.unique():
            type_dic[int(i)] = {"Многофункциональное": 0, "Событийное": 0, "Спортивное": 0, "Учебное": 0}
        for id, row in a.iterrows():
            key = row["id_left"]
            zone_name = row["city_service_type"]
            if zone_name in M:
                type_dic[key]["Многофункциональное"] += 1
            if zone_name in U:
                type_dic[key]["Учебное"] += 1
            if zone_name in S:
                type_dic[key]["Событийное"] += 1
            if zone_name in Sp:
                type_dic[key]["Спортивное"] += 1

        new_dic = {}
        for key, value in type_dic.items():
            max = 0
            max_type = "Многофункциональное"
            for type, v in value.items():
                if v > max:
                    max = v
                    max_type = type
            new_dic[key] = max_type

        for key, value in new_dic.items():
            index = self.OPM[self.OPM["id"] == key].index[0]
            self.OPM.loc[index, "functional_type"] = value
            self.OPM.loc[index, "exists"] = "существующее"

    # Функция строит слой с изхронами для указанного времени
    def get_isohrone_time(self, time):
        street_graph = self.street_graph
        # buildings = gpd.read_file(objects)
        zones = self.OPM

        def nodes_finder(street_graph, objects):
            nodes = ox.distance.nearest_nodes(street_graph, objects.geometry.x, objects.geometry.y)
            return nodes

        def isochrones_generator_time(street_graph, obj_node, DIST):
            subgraph = nx.ego_graph(street_graph, obj_node, radius=DIST, distance="time")
            node_points = [shapely.geometry.Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)]
            isochrones = gpd.GeoSeries(node_points).unary_union.convex_hull
            return isochrones

        zones_points = zones.copy()
        zones_points = zones_points.to_crs(32636)
        zones_points["geometry"] = zones_points["geometry"].centroid
        zones_points["node"] = nodes_finder(street_graph, zones_points)

        zones_iso_time = zones_points
        zones_iso_time.geometry = zones_iso_time.node.apply(lambda x: isochrones_generator_time(street_graph, x, time))
        zones_iso_time = zones_iso_time.to_crs(4326)

        return zones_iso_time

    def create_isochrones(self, street_graph_path):
        self.street_graph = nx.read_graphml(street_graph_path)
        self.iso_20 = self.get_isohrone_time(20)
        self.iso_10 = self.get_isohrone_time(10)
        self.iso_5 = self.get_isohrone_time(5)

    def load_isochrones(self, path):
        self.iso_20 = gpd.read_file(path + "iso_20.geojson")
        self.iso_10 = gpd.read_file(path + "iso_10.geojson")
        self.iso_5 = gpd.read_file(path + "iso_5.geojson")

    def nodes_list(self, iso):
        # функция, которая возвращает словарь, где ключи - это id зон, а значения - список домов, котлорые входят в зону
        a = gpd.sjoin(iso, self.house)
        dic = {}
        for i in a.id.unique():
            dic[int(i)] = []
        for id, row in a.iterrows():
            key = row["id"]
            dic[key].append(row["house_id"])

        return dic

    def add_nodes(self, isochrone, time):
        nodes = self.nodes_list(isochrone)
        for zone, value in nodes.items():
            for house in value:
                self.graph[zone][house]["time"] = time

    def create_graph(self):
        if self.is_graph_exist == False:
            nodes = self.nodes_list(self.iso_20)
            g = nx.Graph(nodes)

            for key in g.nodes():
                if key < 15000:
                    index = self.OPM[self.OPM["id"] == key].index[0]
                    g.nodes[key]["is_zone"] = True
                    g.nodes[key]["area_type"] = self.OPM.loc[index]["area_type"]
                    g.nodes[key]["area"] = self.OPM.loc[index]["area"]
                    g.nodes[key]["functional_type"] = self.OPM.loc[index]["functional_type"]
                    g.nodes[key]["type"] = self.OPM.loc[index]["exists"]
                else:
                    index = self.house[self.house["house_id"] == key].index[0]
                    g.nodes[key]["is_zone"] = False
                    g.nodes[key]["people"] = self.house.loc[index]["people"]
                    g.nodes[key]["provision_ev"] = 0
                    g.nodes[key]["provision_mf"] = 0
                    g.nodes[key]["provision_ed"] = 0
                    g.nodes[key]["provision_sp"] = 0

            for zone, value in nodes.items():
                for house in value:
                    g[zone][house]["time"] = 20

            self.graph = g
            self.add_nodes(self.iso_10, 10)
            self.add_nodes(self.iso_5, 5)

            nodes_list = g.nodes()
            for key in nodes_list:
                node = g.nodes[key]
                if node["is_zone"] == True and node["area_type"] == "small":
                    to_remove = []
                    for house, dist in g[key].items():
                        if dist["time"] == 20 or dist["time"] == 10:
                            # print(dist['time'])
                            to_remove.append(house)
                    for r in to_remove:
                        g.remove_edge(key, r)

                if node["is_zone"] == True and node["area_type"] == "medium":
                    to_remove = []
                    for house, dist in g[key].items():
                        if dist["time"] == 20:
                            to_remove.append(house)
                    for r in to_remove:
                        g.remove_edge(key, r)

            # Подсчет количества людей для каждой зоны
            for node_id in g.nodes():
                if g.nodes[node_id]["is_zone"] == True:
                    people = 0
                    neighbours = g[node_id]
                    for key, value in neighbours.items():
                        people += g.nodes[key]["people"]
                    g.nodes[node_id]["people_around"] = people
            self.is_graph_exist = True

            # Удаление необеспеченных домов
            drop = []
            for n in g:
                neighbors = list(g.neighbors(n))
                if len(neighbors) == 0:
                    drop.append(n)
            g.remove_nodes_from(drop)

            for n in g:
                if g.nodes[n]["is_zone"] == True:
                    neighbor = g.neighbors(n)
                    if g.nodes[n]["type"] == "существующее":
                        self.existed.append(n)
                        if g.nodes[n]["functional_type"] == "Событийное":
                            for neigh in neighbor:
                                g.nodes[neigh]["provision_ev"] += 1
                        if g.nodes[n]["functional_type"] == "Многофункциональное":
                            for neigh in neighbor:
                                g.nodes[neigh]["provision_mf"] += 1
                        if g.nodes[n]["functional_type"] == "Учебное":
                            for neigh in neighbor:
                                g.nodes[neigh]["provision_ed"] += 1
                        if g.nodes[n]["functional_type"] == "Спортивное":
                            for neigh in neighbor:
                                g.nodes[neigh]["provision_sp"] += 1
            self.graph = g
            return self.graph

        else:
            return self.graph

    def OPM_info(self):
        all = 0
        ev = 0
        sp = 0
        mf = 0
        ed = 0
        for n in self.graph:
            if self.graph.nodes[n]["is_zone"] == True and self.graph.nodes[n]["type"] == "существующее":
                all += 1
                if self.graph.nodes[n]["functional_type"] == "Событийное":
                    ev += 1
                if self.graph.nodes[n]["functional_type"] == "Спортивное":
                    sp += 1
                if self.graph.nodes[n]["functional_type"] == "Многофункциональное":
                    mf += 1
                if self.graph.nodes[n]["functional_type"] == "Учебное":
                    ed += 1
        print(f"количество Многофункциональных ОП: {mf}")
        print(f"количество Событийных ОП: {ev}")
        print(f"количество Спортивных ОП: {sp}")
        print(f"количество Учебных ОП: {ed}")
        print(f"общее количество ОП: {all}")

    def provision_info(self):
        houses_all = 0
        houses_covered = 0
        houses_ev = 0
        houses_sp = 0
        houses_mf = 0
        houses_ed = 0
        t = 0
        time = 0
        zone = 0
        for n in self.graph:
            if self.graph.nodes[n]["is_zone"] == False:
                houses_all += 1
                if self.graph.nodes[n]["provision_ev"] >= 1:
                    houses_ev += 1
                if self.graph.nodes[n]["provision_sp"] >= 1:
                    houses_sp += 1
                if self.graph.nodes[n]["provision_mf"] >= 1:
                    houses_mf += 1
                if self.graph.nodes[n]["provision_ed"] >= 1:
                    houses_ed += 1
                if (
                    self.graph.nodes[n]["provision_ev"] >= 1
                    and self.graph.nodes[n]["provision_mf"] >= 1
                    and self.graph.nodes[n]["provision_sp"] >= 1
                    and self.graph.nodes[n]["provision_ed"] >= 1
                ):
                    houses_covered += 1
            if self.graph.nodes[n]["is_zone"] == True and self.graph.nodes[n]["type"] == "существующее":
                zone += 1
                neighbor = list(self.graph.neighbors(n))
                for neigh in neighbor:
                    t += self.graph[n][neigh]["time"]
                time += t / len(neighbor)
                t = 0
        print(f"количество домов, обеспеченных многофункциональным типом: {houses_mf}")
        print(f"количество домов, обеспеченных событийным типом: {houses_ev}")
        print(f"количество домов, обеспеченных спортивным типом: {houses_sp}")
        print(f"количество домов, обеспеченных учебным типом: {houses_ed}")
        print(f"количество полностью обеспеченных домов: {houses_covered}, общее количество домов: {houses_all}")
        time = time / zone
        print(f"Среднее время пути от домов до ОП {round(time, 2)} минуты")

    # функция с суммированием по домам для выбора места для ОПМ при поиске полного решения
    def best_location_full(self, func_type):
        max = 0
        index = -1
        sum = 0
        if func_type == "Событийное":
            prov = "provision_ev"
        if func_type == "Многофункциональное":
            prov = "provision_mf"
        if func_type == "Учебное":
            prov = "provision_ed"
        if func_type == "Спортивное":
            prov = "provision_sp"
        for n in self.graph:
            if (
                self.graph.nodes[n]["is_zone"] == True
                and self.graph.nodes[n]["type"] == "потенциальное"
                and self.graph.nodes[n]["people_around"] * 2 / self.graph.nodes[n]["area"] <= 1
            ):
                neighbor = list(self.graph.neighbors(n))
                for neigh in neighbor:
                    if self.graph.nodes[neigh][prov] == 0:
                        sum += len(neighbor)
                        # sum += 1 / self.graph[n][neigh]['time']
                        # sum += self.graph.nodes[n]['people_around']
                if sum > max:
                    index = n
                    max = sum
                sum = 0
        return max, index, prov

    def best_location1(self, func_type, area_type):
        max = 0
        index = -1
        sum = 0
        if func_type == "Событийное":
            prov = "provision_ev"
        if func_type == "Многофункциональное":
            prov = "provision_mf"
        if func_type == "Учебное":
            prov = "provision_ed"
        if func_type == "Спортивное":
            prov = "provision_sp"
        for n in self.graph:
            if (
                self.graph.nodes[n]["is_zone"] == True
                and self.graph.nodes[n]["type"] == "потенциальное"
                and self.graph.nodes[n]["people_around"] * 2 / self.graph.nodes[n]["area"] <= 1
                and self.graph.nodes[n]["area_type"] == area_type
            ):
                neighbor = list(self.graph.neighbors(n))
                for neigh in neighbor:
                    if self.graph.nodes[neigh][prov] == 0:
                        # sum += gr.nodes[neigh]['people'] / gr[n][neigh]['time']
                        sum += 1 / self.graph[n][neigh]["time"]
                    index = n
                    max = sum
                sum = 0
        return max, index

    # поиск полного решения
    def full_solution(self, max_public_spaces):
        flag = 4
        order = ["Многофункциональное", "Событийное", "Спортивное", "Учебное"]
        num_public_spaces = 0  # Counter for the number of public spaces created

        while True:
            for func_type in order:
                loc = self.best_location_full(func_type)
                if loc[0] != 0 and num_public_spaces < max_public_spaces:
                    self.graph.nodes[loc[1]]["type"] = "существующее"
                    self.graph.nodes[loc[1]]["functional_type"] = func_type
                    neighbor = self.graph.neighbors(loc[1])
                    for neigh in neighbor:
                        self.graph.nodes[neigh][loc[2]] += 1
                    num_public_spaces += 1  # Increment the counter
                else:
                    flag -= 1
            if flag == 0 or num_public_spaces >= max_public_spaces:
                break
            flag = 4

    # локальный поиск, когда нельзя закрывать изначально существующие ОПМ
    def local_search(self):
        i = 0
        closed_func = ""
        nodes = list(self.graph)
        func = {
            "Многофункциональное": "provision_mf",
            "Событийное": "provision_ev",
            "Спортивное": "provision_sp",
            "Учебное": "provision_ed",
        }
        flag = True
        zones_open = len(self.existed)
        ls = zones_open
        clsd = {n: 0 for n in self.graph}
        while i < len(nodes):
            if (
                self.graph.nodes[nodes[i]]["is_zone"] == True
                and self.graph.nodes[nodes[i]]["type"] == "существующее"
                and nodes[i] not in self.existed
            ):
                ls -= 1
                clsd[nodes[i]] -= 1
                closed_func = self.graph.nodes[nodes[i]]["functional_type"]
                self.graph.nodes[nodes[i]]["functional_type"] = ""
                self.graph.nodes[nodes[i]]["type"] = "потенциальное"
                neighbors = list(self.graph.neighbors(nodes[i]))
                for neigh in neighbors:
                    self.graph.nodes[neigh][func[closed_func]] -= 1
                while True:
                    loc = self.best_location_full(closed_func)
                    if loc[0] != 0:
                        ls += 1
                        self.graph.nodes[loc[1]]["type"] = "существующее"
                        self.graph.nodes[loc[1]]["functional_type"] = closed_func
                        clsd[loc[1]] += 1
                        neighbor = list(self.graph.neighbors(loc[1]))
                        for neigh in neighbor:
                            self.graph.nodes[neigh][loc[2]] += 1
                    else:
                        break
                for n in self.graph:
                    if (
                        self.graph.nodes[n]["is_zone"] == True
                        and self.graph.nodes[n]["type"] == "существующее"
                        and self.graph.nodes[n]["functional_type"] == closed_func
                        and n not in self.existed
                    ):
                        neighbors = list(self.graph.neighbors(n))
                        for neigh in neighbors:
                            if self.graph.nodes[neigh][func[closed_func]] == 1:
                                flag = False
                                break
                        if flag:
                            ls -= 1
                            clsd[n] -= 1
                            self.graph.nodes[n]["type"] = "потенциальное"
                            self.graph.nodes[n]["functional_type"] = ""
                            for neigh in neighbors:
                                self.graph.nodes[neigh][func[closed_func]] -= 1
                        flag = True
                if ls >= zones_open:
                    for k in clsd:
                        if clsd[k] == -1:
                            self.graph.nodes[k]["type"] = "существующее"
                            self.graph.nodes[k]["functional_type"] = closed_func
                            neighbor = list(self.graph.neighbors(k))
                            for neigh in neighbor:
                                self.graph.nodes[neigh][func[closed_func]] += 1
                            clsd[k] = 0
                        elif clsd[k] == 1:
                            self.graph.nodes[k]["functional_type"] = ""
                            self.graph.nodes[k]["type"] = "потенциальное"
                            neighbors = list(self.graph.neighbors(k))
                            for neigh in neighbors:
                                self.graph.nodes[neigh][func[closed_func]] -= 1
                            clsd[k] = 0
                    ls = zones_open
                    i += 1
                else:
                    i = 0
                    nodes = list(self.graph)
                    zones_open = ls
                    clsd = {n: 0 for n in self.graph}
            else:
                i += 1

    def provision(self):
        for n in self.graph:
            if self.graph.nodes[n]["is_zone"] == False:
                self.graph.nodes[n]["provision"] = 0
                if self.graph.nodes[n]["provision_mf"] > 0:
                    self.graph.nodes[n]["provision"] += 0.25
                if self.graph.nodes[n]["provision_ev"] > 0:
                    self.graph.nodes[n]["provision"] += 0.25
                if self.graph.nodes[n]["provision_ed"] > 0:
                    self.graph.nodes[n]["provision"] += 0.25
                if self.graph.nodes[n]["provision_sp"] > 0:
                    self.graph.nodes[n]["provision"] += 0.25

    def get_ratio(self, func_type, area):
        r = [[0.53, 0.05, 0.1, 0.17, 0.15], [0.6, 0.1, 0.1, 0.2], [0.67, 0.1, 0.23], [0.6, 0.1, 0.2, 0.1]]
        if func_type == "Многофункциональное":
            a = f"Отдых и общение: {round(r[0][0] * area, 2)} м^2, Обслуживание базовых потребностей: {round(r[0][1] * area, 2)} м^2, Событийная: {round(r[0][2] * area, 2)} м^2, Спортивная: {round(r[0][3] * area, 2)} м^2, Учебная: {round(r[0][4] * area, 2)} м^2"
        if func_type == "Спортивное":
            a = f"Спортивная: {round(r[1][0] * area, 2)} м^2, Обслуживание базовых потребностей: {round(r[1][1] * area, 2)} м^2, Событийная: {round(r[1][2] * area, 2)} м^2, Отдых и общение: {round(r[1][3] * area, 2)} м^2"
        if func_type == "Событийное":
            a = f"Событийная: {round(r[2][0] * area, 2)} м^2, Обслуживание базовых потребностей: {round(r[2][1] * area, 2)} м^2, Отдых и общение: {round(r[2][2] * area, 2)} м^2"
        if func_type == "Учебное":
            a = f"Учебная: {round(r[3][0] * area, 2)} м^2, Событийная: {round(r[3][1] * area, 2)} м^2, Отдых и общение: {round(r[3][2] * area, 2)} м^2, Обслуживание базовых потребностей: {round(r[3][3] * area, 2)} м^2"
        return a

    def recommend_services(self, func_type, area):
        a = self.get_ratio(func_type, area)
        a = a.split(" м^2, ")
        f_zone = [o.split(": ") for o in a]
        f_zone[len(f_zone) - 1][1] = f_zone[len(f_zone) - 1][1].split(" ")[0]
        services = []
        for f in f_zone:
            f[1] = float(f[1])
            if f[0] == "Спортивная":
                if (f[1] >= 200 and f[1] <= 250) or (f[1] > 500 and f[1] <= 800):
                    services.append("Универсальное спортивное поле")
                if f[1] > 250 and f[1] <= 500:
                    services.append("Спортивная площадка")
                if f[1] > 800 and f[1] <= 1200:
                    services.append("Скейтпарк")
                if f[1] > 1200:
                    services.append("Универсальное спортивное поле")
                    services.append("Скейтпарк")
                    services.append("Спортивная площадка")
            if f[0] == "Обслуживание базовых потребностей":
                if f[1] >= 3 and f[1] <= 15:
                    services.append("Общественный туалет")
                if f[1] > 15 and f[1] <= 30:
                    services.append("Фуд-трак")
                if f[1] > 30 and f[1] <= 100:
                    services.append("Кафе")
                if f[1] > 100:
                    services.append("Кафе")
                    services.append("Фуд-трак")
                    services.append("Общественный туалет")
            if f[0] == "Событийная":
                if f[1] >= 20 and f[1] <= 50:
                    services.append("Музыкальное пространство")
                if f[1] > 50 and f[1] <= 100:
                    services.append("Зона искусства")
                if f[1] > 100 and f[1] <= 500:
                    services.append("Сцена (амфитеатр)")
                if f[1] > 500:
                    services.append("Сцена (амфитеатр)")
                    services.append("Зона искусства")
                    services.append("Музыкальное пространство")
            if f[0] == "Отдых и общение":
                if f[1] >= 10 and f[1] <= 100:
                    services.append("Беседки")
                    services.append("Зелёные зоны")
                if f[1] > 100 and f[1] <= 400:
                    services.append("Игровые зоны")
                    services.append("Зелёные зоны")
                if f[1] > 400:
                    services.append("Игровые зоны")
                    services.append("Беседки")
                    services.append("Зелёные зоны")
            if f[0] == "Учебная":
                if f[1] >= 20 and f[1] <= 100:
                    services.append("Учебная зона")
                if f[1] > 100 and f[1] <= 200:
                    services.append("Лектории")
                if f[1] > 200 and f[1] <= 300:
                    services.append("Выставки")
                if f[1] >= 320:
                    services.append("Учебная зона")
                    services.append("Лектории")
                    services.append("Выставки")
        return ", ".join(services)

    def geo(self):
        for n in self.graph:
            if self.graph.nodes[n]["is_zone"] == True and self.graph.nodes[n]["type"] == "существующее":
                indx = self.OPM[self.OPM["id"] == n].index[0]
                self.OPM.loc[indx, "functional_type"] = self.graph.nodes[n]["functional_type"]
        opm = self.OPM[self.OPM["functional_type"] != ""]
        opm = opm[["id", "functional_type", "area_type", "area", "geometry"]]
        opm["ratio"] = ""
        opm["services"] = ""
        for i, row in opm.iterrows():
            opm.loc[i, "ratio"] = self.get_ratio(row["functional_type"], row["area"])
            # opm['services'][i] = self.recommend_services(opm['ratio'][i])
            opm.loc[i, "services"] = self.recommend_services(row["functional_type"], row["area"])
        return opm

    def get_houses(self):
        self.house["provision_ev"] = 0
        self.house["provision_mf"] = 0
        self.house["provision_ed"] = 0
        self.house["provision_sp"] = 0
        self.house["provision"] = 0.0
        self.provision()
        for n in self.graph:
            if self.graph.nodes[n]["is_zone"] == False:
                indx = self.house[self.house["house_id"] == n].index[0]
                self.house.loc[indx, "provision_ev"] = self.graph.nodes[n]["provision_ev"]
                self.house.loc[indx, "provision_mf"] = self.graph.nodes[n]["provision_mf"]
                self.house.loc[indx, "provision_ed"] = self.graph.nodes[n]["provision_ed"]
                self.house.loc[indx, "provision_sp"] = self.graph.nodes[n]["provision_sp"]
                self.house.loc[indx, "provision"] = self.graph.nodes[n]["provision"]
        return self.house

    def draw(self):
        fig = plt.figure()
        fig.set_figheight(50)
        fig.set_figwidth(50)
        nx.draw_kamada_kawai(self.graph, with_labels=False, font_weight="bold", node_size=5, width=0.5)
        plt.show()
