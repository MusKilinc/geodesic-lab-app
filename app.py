import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import eigh
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.graph_objects as go
import plotly.express as px


# ---------------------------------------------
# Yardımcı Fonksiyon: Geodezik Kubbe Tipini Belirle
# ---------------------------------------------
def generate_dome_geometry(dome):
    dome_type = dome.get("type")
    if dome_type == 1:
        return tetrahedron(dome)
    elif dome_type == 2:
        return octahedron(dome)
    elif dome_type == 3:
        return icosahedron(dome)
    else:
        raise ValueError("Geçersiz kubbe tipi. 1=Tetrahedron, 2=Octahedron, 3=Icosahedron")

# ---------------------------------------------
# Geodezik Kubbe Ana Fonksiyon Akışı
# ---------------------------------------------
def run_geodesic_analysis(dome):
    dome = generate_dome_geometry(dome)                 # Geometri oluştur
    dome = crtruss(dome)                                # Yapı bilgileri
    dome = grpdet(dome)                                 # Grup bilgileri
    dome = define_supports(dome)                        # Mesnet bilgileri
    if dome.get("apply_deadload", True):
        dome = deadload(dome)                           # Ölü yük (isteğe bağlı)
    dome = dispforce(dome)                              # Dış yükler
    dome = create_global_stiffness_data(dome)           # Küresel rijitlik bilgileri + kütle
    dome = analysis(dome)                               # Yapısal analiz
    dome = compute_dynamic_parameters(dome)             # Dinamik parametreler

    # Bilgi özeti
    lengths = [m["length"] for m in dome["members"]]
    weights = [m["weight"] for m in dome["members"]]
    dome_type_name = {1: "Tetrahedron", 2: "Octahedron", 3: "Icosahedron"}.get(dome["type"], "Unknown")

    dome["info"] = {
        "type": dome_type_name,
        "span": dome.get("span"),
        "height": dome.get("height"),
        "freq": dome.get("freq"),
        "node_number": len(dome["nodes"]),
        "member_number": len(dome["members"]),
        "group_number": len(dome.get("groups", {})),
        "longest_member": {
            "index": int(np.argmax(lengths)) + 1,
            "length": float(np.max(lengths))
        },
        "shortest_member": {
            "index": int(np.argmin(lengths)) + 1,
            "length": float(np.min(lengths))
        },
        "total_length": float(np.sum(lengths)),
        "total_weight": float(np.sum(weights))
    }

    # Ek sonuç bilgileriyle zenginleştir
    dome = enrich_dome_info_with_results(dome)

    # Gereksiz tekrarlar temizleniyor
    for key in ["type", "span", "height", "freq", "nodenum", "memnum", "support", "longest_member", "shortest_member"]:
        dome.pop(key, None)

    return dome

def enrich_dome_info_with_results(dome):
    # Gerilmeleri çekiyoruz
    stresses = np.array([m["stress"] for m in dome["members"]])
    tensile_idx = int(np.argmax(stresses))
    compressive_idx = int(np.argmin(stresses))

    # Deplasman vektörleri
    disps = dome["displacement"]
    ux_max_idx = int(np.argmax(np.abs(disps[:, 0])))
    uy_max_idx = int(np.argmax(np.abs(disps[:, 1])))
    uz_max_idx = int(np.argmax(np.abs(disps[:, 2])))

    # Güncelleme
    dome["info"].update({
        "max_tension": {
            "member_id": dome["members"][tensile_idx]["id"],
            "stress": float(stresses[tensile_idx])
        },
        "max_compression": {
            "member_id": dome["members"][compressive_idx]["id"],
            "stress": float(stresses[compressive_idx])
        },
        "max_displacement": {
            "X": {
                "node_id": ux_max_idx + 1,
                "value": float(disps[ux_max_idx, 0])
            },
            "Y": {
                "node_id": uy_max_idx + 1,
                "value": float(disps[uy_max_idx, 1])
            },
            "Z": {
                "node_id": uz_max_idx + 1,
                "value": float(disps[uz_max_idx, 2])
            }
        }
    })

    return dome

# ---------------------------------------------
# Tetrahedron Geometrisi
# ---------------------------------------------
def tetrahedron(dome):
    sp = dome["span"]
    ht = dome["height"]
    fr = dome["freq"]
    face = 3

    # Yarıçap ve açı hesapları
    R = ((sp / 2) ** 2 / ht + ht) / 2
    if ht > R:
        raise ValueError("Kubbe yüksekliği yarıçaptan büyük olamaz.")
    alpha = np.degrees(np.arcsin(sp / (2 * R)))
    phi = alpha / fr

    dome["radius"] = R
    dome["angle"] = alpha
    dome["fr_angle"] = phi

    ds = sum(range(1, fr+1)) * face + 1
    dome["nodenum"] = ds
    nodes = np.zeros((ds, 4))

    es = (sum(range(1, fr+1)) * 3 - fr) * face
    dome["memnum"] = es
    eleman = np.zeros((es, 2), dtype=int)
    mem = np.zeros((es, 5))

    # Tepe noktası
    nodes[0, :] = [1, 0, 0, R]
    dx = [nodes[0:1, :]]

    j = 0
    for i in range(1, fr+1):
        rx = R * np.sin(np.radians(phi * i))
        ar = 2 * np.pi / (face * i)
        aci = np.arange(0, 2 * np.pi, ar)
        ll = np.arange(j+1, j+1+len(aci))
        nodes[ll, 0] = ll + 1
        nodes[ll, 1] = rx * np.cos(aci)
        nodes[ll, 2] = rx * np.sin(aci)
        nodes[ll, 3] = R * np.cos(np.radians(phi * i))
        j += len(aci)
        dx.append(nodes[ll, :])

    dome["nodes"] = nodes

    # Eleman oluşturma
    index = 0
    d1 = int(dx[0][0, 0]) - 1
    d2 = (dx[1][:, 0] - 1).astype(int)

    for i in range(len(dx[1])):
        eleman[index, :] = [d2[i], d1]
        index += 1

    for i in range(1, fr):
        d2 = (dx[i+1][::i+1, 0] - 1).astype(int)
        d1 = (dx[i][::i, 0] - 1).astype(int)
        for j in range(len(d1)):
            eleman[index, :] = [d1[j], d2[j]]
            index += 1

    for i in range(1, fr+1):
        kd = (dx[i][:, 0] - 1).astype(int)
        for j in range(len(kd)-1):
            eleman[index, :] = [kd[j], kd[j+1]]
            index += 1
        eleman[index, :] = [kd[-1], kd[0]]
        index += 1

    ii = fr+1
    for i in range(1, fr):
        d2 = (dx[i+1][:, 0] - 1).astype(int)
        d1 = (dx[i][:, 0] - 1).astype(int)
        d2x = (dx[i+1][::i+1, 0] - 1).astype(int)
        d1x = (dx[i][::i, 0] - 1).astype(int)
        tmp = np.setdiff1d(d2, d2x)
        tmp = np.hstack([tmp[-1], tmp])
        for j in range(len(d1)):
            eleman[index, :] = [d1[j], tmp[j]]
            eleman[index+1, :] = [d1[j], tmp[j+1]]
            index += 2

    mem[:, 0] = np.arange(1, len(eleman)+1)
    mem[:, 1:3] = eleman + 1
    mem[:, 3] = dome["geomat"][0]
    mem[:, 4] = dome["geomat"][1]
    dome["members"] = mem

    supp = dx[-1].copy()
    supp[:, 1:4] = 1
    dome["support"] = supp.astype(int)

    return dome

# ---------------------------------------------
# Icosahedron Geometrisi
# ---------------------------------------------
def icosahedron(dome):
    # giriş parametreleri
    sp = dome["span"]
    ht = dome["height"]
    fr = dome["freq"]
    face = 5

    # Kubbe yarıçap hesaplama
    R = ((sp/2)**2/ht + ht)/2
    if ht > R:
        raise ValueError('height of dome should not be higher than half of span')

    dome["radius"] = R
    alpha = np.degrees(np.arcsin((sp/2)/R))
    dome["angle"] = alpha

    phi = alpha/fr
    dome["fr_angle"] = phi

    ds = sum(range(1, fr+1))*face + 1
    dome["nodenum"] = ds
    node = np.zeros((ds, 4))

    es = (sum(range(1, fr+1))*3 - fr)*face
    dome["memnum"] = es
    eleman = np.zeros((es, 2), dtype=int)
    mem = np.zeros((es, 5))

    # Tepe noktası
    node[0, :] = [1, 0, 0, R]
    dx = [node[0:1, :]]

    j = 0  # Python'da index 0'dan başlar
    for i in range(1, fr+1):
        rx = R * np.sin(np.radians(phi*i))
        ar = 2 * np.pi / (face * i)
        aci = np.arange(0, 2*np.pi, ar)
        ll = np.arange(j+1, j+1+len(aci))
        
        node[ll, 0] = ll + 1
        node[ll, 1] = rx * np.cos(aci)
        node[ll, 2] = rx * np.sin(aci)
        node[ll, 3] = R * np.cos(np.radians(phi*i))
        
        j += len(aci)
        dx.append(node[ll, :])

    dome["nodes"] = node

    # Elemanları oluşturma (üyeler)
    index = 0
    d1 = int(dx[0][0, 0]) - 1
    d2 = (dx[1][:, 0] - 1).astype(int)
    gr = [[] for _ in range(2*fr)]

    for i in range(len(dx[1])):
        eleman[index, :] = [d2[i], d1]
        gr[0].append(index)
        index += 1

    # dikey elemanlar
    for i in range(1, fr):
        d2 = (dx[i+1][::i+1, 0] - 1).astype(int)
        d1 = (dx[i][::i, 0] - 1).astype(int)
        for j in range(len(d1)):
            eleman[index, :] = [d1[j], d2[j]]
            gr[0].append(index)
            index += 1

    # yatay elemanlar
    for i in range(1, fr+1):
        kd = (dx[i][:, 0] - 1).astype(int)
        gr[i] = []
        for j in range(len(kd)-1):
            eleman[index, :] = [kd[j], kd[j+1]]
            gr[i].append(index)
            index += 1
        eleman[index, :] = [kd[-1], kd[0]]
        index += 1

    # çapraz elemanlar
    ii = fr+1
    for i in range(1, fr):
        ii += 1
        d2 = (dx[i+1][:, 0] - 1).astype(int)
        d1 = (dx[i][:, 0] - 1).astype(int)
        d2x = (dx[i+1][::i+1, 0] - 1).astype(int)
        d1x = (dx[i][::i, 0] - 1).astype(int)
        tmp = np.setdiff1d(d2, d2x)
        tmp = np.hstack([tmp[-1], tmp])
        
        gr.append([])
        for j in range(len(d1)):
            eleman[index, :] = [d1[j], tmp[j]]
            eleman[index+1, :] = [d1[j], tmp[j+1]]
            gr[ii-1].extend([index, index+1])
            index += 2

    mem[:, 0] = np.arange(1, len(eleman)+1)
    mem[:, 1:3] = eleman + 1
    mem[:, 3] = dome["geomat"][0]
    mem[:, 4] = dome["geomat"][1]
    dome["members"] = mem.astype(int)

    supp = dx[-1].copy()
    supp[:, 1:4] = 1
    dome["support"] = supp.astype(int)

    return dome

# ---------------------------------------------
# Octahedron Geometrisi
# ---------------------------------------------
def octahedron(dome):
    sp = dome["span"]
    ht = dome["height"]
    fr = dome["freq"]
    face = 4  # Octahedron için

    # Yarıçap ve açı hesapları
    R = ((sp / 2) ** 2 / ht + ht) / 2
    if ht > R:
        raise ValueError("Kubbe yüksekliği yarıçaptan büyük olamaz.")
    alpha = np.degrees(np.arcsin(sp / (2 * R)))
    phi = alpha / fr

    dome["radius"] = R
    dome["angle"] = alpha
    dome["fr_angle"] = phi

    ds = sum(range(1, fr+1)) * face + 1
    dome["nodenum"] = ds
    nodes = np.zeros((ds, 4))

    es = (sum(range(1, fr+1)) * 3 - fr) * face
    dome["memnum"] = es
    eleman = np.zeros((es, 2), dtype=int)
    mem = np.zeros((es, 5))

    # Tepe noktası
    nodes[0, :] = [1, 0, 0, R]
    dx = [nodes[0:1, :]]

    j = 0
    for i in range(1, fr+1):
        rx = R * np.sin(np.radians(phi * i))
        ar = 2 * np.pi / (face * i)
        aci = np.arange(0, 2 * np.pi, ar)
        ll = np.arange(j+1, j+1+len(aci))
        nodes[ll, 0] = ll + 1
        nodes[ll, 1] = rx * np.cos(aci)
        nodes[ll, 2] = rx * np.sin(aci)
        nodes[ll, 3] = R * np.cos(np.radians(phi * i))
        j += len(aci)
        dx.append(nodes[ll, :])

    dome["nodes"] = nodes

    # Eleman oluşturma
    index = 0
    d1 = int(dx[0][0, 0]) - 1
    d2 = (dx[1][:, 0] - 1).astype(int)

    for i in range(len(dx[1])):
        eleman[index, :] = [d2[i], d1]
        index += 1

    for i in range(1, fr):
        d2 = (dx[i+1][::i+1, 0] - 1).astype(int)
        d1 = (dx[i][::i, 0] - 1).astype(int)
        for j in range(len(d1)):
            eleman[index, :] = [d1[j], d2[j]]
            index += 1

    for i in range(1, fr+1):
        kd = (dx[i][:, 0] - 1).astype(int)
        for j in range(len(kd)-1):
            eleman[index, :] = [kd[j], kd[j+1]]
            index += 1
        eleman[index, :] = [kd[-1], kd[0]]
        index += 1

    ii = fr+1
    for i in range(1, fr):
        d2 = (dx[i+1][:, 0] - 1).astype(int)
        d1 = (dx[i][:, 0] - 1).astype(int)
        d2x = (dx[i+1][::i+1, 0] - 1).astype(int)
        d1x = (dx[i][::i, 0] - 1).astype(int)
        tmp = np.setdiff1d(d2, d2x)
        tmp = np.hstack([tmp[-1], tmp])
        for j in range(len(d1)):
            eleman[index, :] = [d1[j], tmp[j]]
            eleman[index+1, :] = [d1[j], tmp[j+1]]
            index += 2

    mem[:, 0] = np.arange(1, len(eleman)+1)
    mem[:, 1:3] = eleman + 1
    mem[:, 3] = dome["geomat"][0]
    mem[:, 4] = dome["geomat"][1]
    dome["members"] = mem

    supp = dx[-1].copy()
    supp[:, 1:4] = 1
    dome["support"] = supp.astype(int)

    return dome

# ---------------------------------------------
# Eleman Bilgileri ve Rijitlik/Kütle Verisi Oluşturma
# ---------------------------------------------
def crtruss(dome):
    nodes = dome["nodes"]
    raw_members = dome["members"]
    density = dome["geomat"][2]

    members = []

    for i, m in enumerate(raw_members):
        start_node = int(m[1]) - 1
        end_node = int(m[2]) - 1

        x1, y1, z1 = nodes[start_node, 1:4]
        x2, y2, z2 = nodes[end_node, 1:4]

        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        cx, cy, cz = (x2 - x1) / L, (y2 - y1) / L, (z2 - z1) / L

        A = m[3]
        E = m[4]

        volume = A * L
        weight = volume * density * 9.81
        mass = volume * density

        dof = np.array([
            3 * start_node, 3 * start_node + 1, 3 * start_node + 2,
            3 * end_node, 3 * end_node + 1, 3 * end_node + 2
        ])

        k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
        T = np.zeros((2, 6))
        T[0, :3] = [cx, cy, cz]
        T[1, 3:] = [cx, cy, cz]
        k_global = T.T @ k_local @ T

        # Lumped mass matrix (6x6)
        m_lumped = (mass / 2) * np.eye(6)

        members.append({
            "id": i + 1,
            "start_node": start_node + 1,
            "end_node": end_node + 1,
            "length": L,
            "cx": cx,
            "cy": cy,
            "cz": cz,
            "Area": A,
            "Elastisite": E,
            "weight": weight,
            "mass": mass,
            "dof": dof,
            "k_local": k_local,
            "T": T,
            "k_global": k_global,
            "m_global": m_lumped,
            "internal_force": 0,
            "stress": 0
        })

    dome["members"] = members
    return dome

# ---------------------------------------------
# Grup Bilgisi Oluşturma (Uzunluklara Göre)
# ---------------------------------------------
def grpdet(dome, tol=1e-3):
    members = dome["members"]
    unique_lengths = []
    groups = []

    for m in members:
        L = m["length"]
        found = False
        for i, ul in enumerate(unique_lengths):
            if np.isclose(L, ul, atol=tol):
                groups[i].append(m["id"])
                found = True
                break
        if not found:
            unique_lengths.append(L)
            groups.append([m["id"]])

    dome["groups"] = {
        f"group_{i+1}": {
            "members": g,
            "length": unique_lengths[i]
        } for i, g in enumerate(groups)
    }
    return dome

# ---------------------------------------------
# Mesnetleri Tanımlama: Taban Düğüm Noktaları
# ---------------------------------------------
def define_supports(dome, z_tol=1e-3):
    nodes = dome["nodes"]
    supports = []

    for node in nodes:
        node_id = int(node[0])
        z = node[3]
        if np.isclose(z, 0.0, atol=z_tol):
            supports.append([node_id, 1, 1, 1])  # X, Y, Z yönünde sabit

    dome["supports"] = np.array(supports)
    return dome

# ---------------------------------------------
# Ölü Yük Hesabı
# ---------------------------------------------
def deadload(dome):
    nodenum = len(dome["nodes"])
    loads = np.zeros((nodenum, 4))  # [node_id, Fx, Fy, Fz]
    loads[:, 0] = np.arange(1, nodenum + 1)

    for i, member in enumerate(dome["members"]):
        n1, n2 = member["start_node"] - 1, member["end_node"] - 1
        w = member["weight"] / 2  # her düğüme yarısı
        loads[n1, 3] -= w
        loads[n2, 3] -= w

    dome["external_forces"] = loads
    return dome

# ---------------------------------------------
# Dış Yüklerin Uygulanması (Yatay, Düşey, Noktasal)
# ---------------------------------------------
def dispforce(dome):
    loads = dome.get("external_forces", np.zeros((len(dome["nodes"]), 4)))
    nodenum = len(dome["nodes"])

    if dome.get("horizontal_load", 0) != 0:
        for i in range(nodenum):
            loads[i, 1] += dome["horizontal_load"]

    if dome.get("vertical_load", 0) != 0:
        for i in range(nodenum):
            loads[i, 3] += dome["vertical_load"]

    if "pointload" in dome:
        for load in dome["pointload"]:
            node, fx, fy, fz = load
            loads[int(node)-1, 1] += fx
            loads[int(node)-1, 2] += fy
            loads[int(node)-1, 3] += fz

    dome["external_forces"] = loads
    return dome

# ---------------------------------------------
# Küresel Rijitlik ve Kütle Matrisi Oluşturma
# ---------------------------------------------
def create_global_stiffness_data(dome):
    dof_total = len(dome["nodes"]) * 3
    K_global = np.zeros((dof_total, dof_total))
    M_global = np.zeros((dof_total, dof_total))

    for m in dome["members"]:
        dof = m["dof"]
        K_global[np.ix_(dof, dof)] += m["k_global"]
        M_global[np.ix_(dof, dof)] += m["m_global"]

    dome["matrices"] = {
        "K_global": K_global,
        "M_global": M_global
    }
    return dome

# ---------------------------------------------
# Statik Analiz Fonksiyonu
# ---------------------------------------------
def analysis(dome):
    nodes = dome["nodes"]
    dof_total = len(nodes) * 3
    K_global = dome["matrices"]["K_global"]
    F_global = np.zeros(dof_total)

    for i, force in enumerate(dome["external_forces"]):
        F_global[i*3:i*3+3] = force[1:4]

    fixed_dofs = []
    for support in dome["supports"]:
        node_id = int(support[0]) - 1
        for j in range(3):
            if support[j+1]:
                fixed_dofs.append(node_id * 3 + j)

    free_dofs = np.setdiff1d(np.arange(dof_total), fixed_dofs)

    U = np.zeros(dof_total)
    U[free_dofs] = np.linalg.solve(K_global[np.ix_(free_dofs, free_dofs)], F_global[free_dofs])

    F_reactions = K_global @ U
    dome["displacement"] = U.reshape(-1, 3)
    dome["reactions_full"] = F_reactions.reshape(-1, 3)
    dome["fixed_dofs"] = fixed_dofs

    # Eleman iç kuvvet ve gerilme hesapları
    for m in dome["members"]:
        dof = m["dof"]
        u_elem = U[dof]
        local_disp = m["T"] @ u_elem
        internal_force = m["k_local"] @ local_disp
        stress = internal_force[1] / m["Area"]
        m["internal_force"] = internal_force[1]
        m["stress"] = stress

    return dome

# ---------------------------------------------
# Dinamik Analiz Fonksiyonu
# ---------------------------------------------
def compute_dynamic_parameters(dome, num_modes=10):
    K_global = dome["matrices"]["K_global"]
    M_global = dome["matrices"]["M_global"]
    dof_total = len(dome["nodes"]) * 3

    fixed_dofs = dome["fixed_dofs"] if isinstance(dome["fixed_dofs"], (list, np.ndarray)) else list(dome["fixed_dofs"])
    free_dofs = np.setdiff1d(np.arange(dof_total), fixed_dofs)

    Kf = K_global[np.ix_(free_dofs, free_dofs)]
    Mf = M_global[np.ix_(free_dofs, free_dofs)]

    eigvals, eigvecs = eigh(Kf, Mf)
    eigvals = np.maximum(eigvals, 0)

    omegas = np.sqrt(eigvals)
    freqs = omegas / (2 * np.pi)
    periods = np.divide(1, freqs, out=np.zeros_like(freqs), where=freqs != 0)

    dome["dynamic"] = {
        "frequencies": freqs[:num_modes],
        "periods": periods[:num_modes],
        "mode_shapes": eigvecs[:, :num_modes]
    }
    return dome
# ---------------------------------------------
# Görselleştirme Fonksiyonu (Plot)
# ---------------------------------------------
def plot_dome(dome, scale=1.0, show_labels=False, label_type=None):
    nodes = dome["nodes"]
    members = dome["members"]
    groups = dome.get("groups", {})
    displacements = dome.get("displacement", np.zeros_like(nodes[:, 1:4])) * scale

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    color_map = cm.get_cmap('tab20', len(groups))
    group_map = {}
    for i, (gname, gdata) in enumerate(groups.items()):
        group_map.update({mid: i for mid in gdata["members"]})

    for i, m in enumerate(members):
        n1, n2 = int(m["start_node"]) - 1, int(m["end_node"]) - 1
        x = [nodes[n1, 1], nodes[n2, 1]]
        y = [nodes[n1, 2], nodes[n2, 2]]
        z = [nodes[n1, 3], nodes[n2, 3]]

        dx = [x[0] + displacements[n1, 0], x[1] + displacements[n2, 0]]
        dy = [y[0] + displacements[n1, 1], y[1] + displacements[n2, 1]]
        dz = [z[0] + displacements[n1, 2], z[1] + displacements[n2, 2]]

        group_index = group_map.get(m["id"], 0)
        ax.plot(dx, dy, dz, color=color_map(group_index), linewidth=1.5)
        
        if show_labels:
            if label_type == "group":
                ax.text(np.mean(dx), np.mean(dy), np.mean(dz), f"G{group_index+1}", fontsize=8)
            elif label_type == "member":
                ax.text(np.mean(dx), np.mean(dy), np.mean(dz), f"M{m['id']}", fontsize=8)
            elif label_type == "stress":
                ax.text(np.mean(dx), np.mean(dy), np.mean(dz), f"{m['stress']:.2f}", fontsize=8)
            elif label_type == "force":
                ax.text(np.mean(dx), np.mean(dy), np.mean(dz), f"{m['internal_force']:.2f}", fontsize=8)
            elif label_type == "length":
                ax.text(np.mean(dx), np.mean(dy), np.mean(dz), f"{m['length']:.1f}", fontsize=8)

    ax.grid(False)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=45)  # Yükseklik ve açı ayarı
    
        # Eksenleri eşitle
    all_x = nodes[:, 1] + displacements[:, 0]
    all_y = nodes[:, 2] + displacements[:, 1]
    all_z = nodes[:, 3] + displacements[:, 2]

    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    z_range = all_z.max() - all_z.min()
    max_range = max(x_range, y_range, z_range)

    x_mid = (all_x.max() + all_x.min()) / 2
    y_mid = (all_y.max() + all_y.min()) / 2
    z_mid = (all_z.max() + all_z.min()) / 2

    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)

    plt.title("Geodesic Dome")
    plt.tight_layout()
    plt.show()

     # Arayüz için görsel döndürme
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()  # Arayüzde üst üste binmemesi için kapat
    return buf

def plot_dome_plotly(dome, scale=1.0, label_type=None):
    nodes = dome["nodes"]
    members = dome["members"]
    groups = dome.get("groups", {})
    displacements = dome.get("displacement", np.zeros_like(nodes[:, 1:4])) * scale

    fig = go.Figure()

    # Grup renk haritası
    colors = px.colors.qualitative.Plotly
    group_map = {}
    for i, (gname, gdata) in enumerate(groups.items()):
        for mid in gdata["members"]:
            group_map[mid] = i

    for m in members:
        n1 = int(m["start_node"]) - 1
        n2 = int(m["end_node"]) - 1

        x = [nodes[n1][1], nodes[n2][1]]
        y = [nodes[n1][2], nodes[n2][2]]
        z = [nodes[n1][3], nodes[n2][3]]

        dx = [x[0] + displacements[n1, 0], x[1] + displacements[n2, 0]]
        dy = [y[0] + displacements[n1, 1], y[1] + displacements[n2, 1]]
        dz = [z[0] + displacements[n1, 2], z[1] + displacements[n2, 2]]

        group_index = group_map.get(m["id"], 0)
        color = colors[group_index % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=dx, y=dy, z=dz,
            mode="lines",
            line=dict(color=color, width=4),
            name=f"G{group_index + 1}",
            showlegend=False
        ))

        if label_type:
            # Etiket koordinatı (orta nokta)
            mx, my, mz = np.mean(dx), np.mean(dy), np.mean(dz)
            if label_type == "group":
                label = f"G{group_index + 1}"
            elif label_type == "member":
                label = f"M{m['id']}"
            elif label_type == "stress":
                label = f"{m['stress']:.2f}"
            elif label_type == "force":
                label = f"{m['internal_force']:.2f}"
            elif label_type == "length":
                label = f"{m['length']:.1f}"
            else:
                label = ""

            fig.add_trace(go.Scatter3d(
                x=[mx], y=[my], z=[mz],
                mode='text',
                text=[label],
                textposition='top center',
                showlegend=False
            ))

    all_x = nodes[:, 1] + displacements[:, 0]
    all_y = nodes[:, 2] + displacements[:, 1]
    all_z = nodes[:, 3] + displacements[:, 2]

    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    z_range = all_z.max() - all_z.min()
    max_range = max(x_range, y_range, z_range)

    x_mid = (all_x.max() + all_x.min()) / 2
    y_mid = (all_y.max() + all_y.min()) / 2
    z_mid = (all_z.max() + all_z.min()) / 2

    scene_config = dict(
        xaxis=dict(range=[x_mid - max_range/2, x_mid + max_range/2], visible=False),
        yaxis=dict(range=[y_mid - max_range/2, y_mid + max_range/2], visible=False),
        zaxis=dict(range=[z_mid - max_range/2, z_mid + max_range/2], visible=False),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1),
        bgcolor="white"
    )

    fig.update_layout(
        title="Geodesic Dome",
        scene=scene_config,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


# app.py
import streamlit as st

st.title("🌐 Geodesic Dome Analysis App")

st.sidebar.header("📐 Dome Parameters")
dome_type = st.sidebar.selectbox("Dome Type", options=["Tetrahedron", "Octahedron", "Icosahedron"])
type_map = {"Tetrahedron": 1, "Octahedron": 2, "Icosahedron": 3}
span = st.sidebar.number_input("Span [mm]", value=500)
height = st.sidebar.number_input("Height [mm]", value=250)
freq = st.sidebar.number_input("Frequency (division)", value=2)

st.sidebar.header("🔩 Material Properties")
E = st.sidebar.number_input("Elastic Modulus [MPa]", value=200000)
A = st.sidebar.number_input("Cross Section Area [mm²]", value=100)
rho = st.sidebar.number_input("Density [kg/m³]", value=7850)

st.sidebar.header("⚙️ Load Options")
apply_deadload = st.sidebar.checkbox("Apply Dead Load", value=True)
ext_horizontal = st.sidebar.number_input("Horizontal Load [N]", value=0)
ext_vertical = st.sidebar.number_input("Vertical Load [N]", value=0)
pointload = st.sidebar.text_input("Point Load (Format: node_id, Fx, Fy, Fz)", "1, 0, 0, -500")

if st.button("🧠 Run Analysis"):
    try:
        pointload_parsed = [list(map(float, p.strip().split(","))) for p in pointload.split(";")]
    except:
        st.error("Point load format is incorrect.")
        pointload_parsed = []

    dome_input = {
        "type": type_map[dome_type],
        "span": span,
        "height": height,
        "freq": freq,
        "geomat": [A, E, rho],
        "apply_deadload": apply_deadload,
        "ext_horizontal": ext_horizontal,
        "ext_vertical": ext_vertical,
        "pointload": pointload_parsed
    }

    with st.spinner("Running structural and dynamic analysis..."):
        dome = run_geodesic_analysis(dome_input)
        st.success("Analysis completed ✅")

        st.subheader("📊 Basic Info")
        st.json(dome["info"])

        st.subheader("📈 Dynamic Properties")
        st.write("Frequencies (Hz):", dome["dynamic"]["frequencies"])
        st.write("Periods (s):", dome["dynamic"]["periods"])

        st.subheader("🖼️ Visualize Dome")
        img_buffer = plot_dome(dome, scale=0, show_labels=True, label_type="group")
        st.image(img_buffer, caption="Geodesic Dome", use_container_width=True)

        st.download_button("📥 Download Dome Data", data=str(dome), file_name="dome_data.txt")
        
        st.plotly_chart(plot_dome_plotly(dome, scale=0, label_type="group"), use_container_width=True)

