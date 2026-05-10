from numpy import array
from pathlib import Path
from numpy.linalg import lstsq
from pandas import DataFrame, Series
from scipy.optimize import curve_fit
from pandas import DataFrame, to_numeric, read_csv
from numpy import array, linspace, exp, log, diag, sqrt, clip, newaxis
from matplotlib.pyplot import errorbar, gca, figure, title, xlabel, ylabel, legend, grid, show

EXTENSION = ".CSV"

def filtrar_extension(directorio, extension = EXTENSION):
    try:
        ruta = Path(directorio)
        patron = f"*{extension}"
        archivos = [archivo.name.removesuffix(EXTENSION) for archivo in ruta.glob(patron) if archivo.is_file()]
        if not archivos:
            raise Exception(f"No se encontraron archivos con la extensión '{extension}' en el directorio '{directorio}'.")
        return archivos
    except FileNotFoundError:
        print(f"Error: El directorio '{directorio}' no existe o la ruta es incorrecta.")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def leer(directorio, skiprows=0, header=None):
    try:
        return read_csv(directorio, skiprows=skiprows, header=header)
    except FileNotFoundError:
        print(f"Error: El archivo '{directorio}' no fue encontrado.")
        return DataFrame()
    except Exception as e:
        print(f"Ocurrio un error al leer el archivo CSV: {e}")
        return DataFrame()
    
def limpiar(directorio, col_t_1=0, col_y_1=1, col_y_2=2):
    try:
        df = leer(directorio)
        columnas_a_limpiar = [col_t_1, col_y_1, col_y_2]
        for col in columnas_a_limpiar:
            df[col] = to_numeric(df[col], errors="coerce")
        df_limpio = df.dropna(subset=columnas_a_limpiar).copy()
        if col_t_1 in df_limpio.columns:
            df_limpio[col_t_1] = (df_limpio[col_t_1] - df_limpio[col_t_1].iloc[0])
        return df_limpio
    except FileNotFoundError:
        print(f"El archivo '{directorio}' no fue encontrado.")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def obtener_datos(nombre_archivo, v_offset, v_ventana, col_t=0, col_va=1, col_vb=2):
    df: DataFrame = limpiar(f"{nombre_archivo}{EXTENSION}", col_t_1=col_t, col_y_1=col_va, col_y_2=col_vb)
    li = df.iloc[:, col_vb].max() - v_ventana
    return df.iloc[:, col_t], df.iloc[:, col_va], df.iloc[:, col_vb] + v_offset, li

def filtrar_datos(x, y, li=None, ls=None):
    mascara = Series([True] * len(y), index=y.index)
    if li is not None:
        mascara &= (y >= li)
    if ls is not None:
        mascara &= (y <= ls)
    return x[mascara], y[mascara]

def valores_representativos(x, y, num_valores=20):
    x_i = x.min()
    x_s = x.max()
    dx = (x_s - x_i) / num_valores
    x_umbral, y_promedio, y_std = [], [], []
    for j in range(num_valores):
        bin_start = x_i + j * dx
        bin_end = x_i + (j + 1) * dx
        y_cercano = y[(x >= bin_start) & (x < bin_end)]
        if not y_cercano.empty:
            x_umbral.append((bin_start + bin_end) / 2)
            y_promedio.append(y_cercano.mean())
            std_val = y_cercano.std()
            y_std.append(std_val if std_val > 0 else 1e-6)
    return array(x_umbral), array(y_promedio), array(y_std)

def regresion_emergencial(x, y, a, c, y_std=None, resolucion_regresion=1000):
    x_fit = linspace(x.min(), x.max(), resolucion_regresion)
    y_log = log(((y - c) / a) + 1)
    w = (y_std - c + a) / y if y_std is not None else None
    x_w = (x * w)[:, newaxis]   # Matriz columna de X pesada
    y_w = y_log * w             # Vector Y pesado
    b = lstsq(x_w, y_w, rcond=None)[0][0]
    y_fit = a * (exp(b * x_fit) - 1) + c
    return ([x_fit, y_fit], b)

def regresion_no_emergencial(x, y, a, b, c, y_std=None, resolucion_regresion=1000):
    def modelo(x, a, b, c):
        return a * (exp(clip(b * x, -700, 700)) - 1) + c
    x_fit = linspace(x.min(), x.max(), resolucion_regresion)
    (a_, b_, c_), pcov = curve_fit(modelo, x, y, [a, b, c], y_std, True, method="lm")
    return ([x_fit, modelo(x_fit, a_, b_, c_)], [a_, b_, c_], sqrt(diag(pcov)))

def graficar_columnas(x, y, tit, nom_x, nom_y, label, markersize=0.1, fmt="-", alpha=1.0, yerr=None, names=None):
    figure(figsize=(10, 5))
    if not isinstance(markersize, list):
        markersize = [markersize] * len(x)
    if not isinstance(fmt, list):
        fmt = [fmt] * len(x)
    if not isinstance(alpha, list):
        alpha = [alpha] * len(x)
    if not isinstance(yerr, list):
        yerr = [yerr] * len(x)
    if not isinstance(names, list):
        names = [names] * len(x)
    for x_i, y_i, label_i, markersize_i, fmt_i, alpha_i, yerr_i, names_i in zip(x, y, label, markersize, fmt, alpha, yerr, names):
        x_i = array(x_i)
        y_i = array(y_i)
        errorbar(x=x_i, y=y_i, label=label_i, markersize=markersize_i, fmt=fmt_i, ecolor="black", elinewidth=0.5, alpha=alpha_i, yerr=yerr_i, capsize=3)
        ax = gca()
        if names_i is not None:
            for x_j, y_j, names_j in zip(x_i, y_i, names_i):
                ax.annotate(names_j, xy=(x_j, y_j), xytext=(0, 2), textcoords="offset points", fontsize=8, ha="center", va="bottom")
    title(tit)
    xlabel(nom_x)
    ylabel(nom_y)
    legend()
    grid()
    show()

def graficar_voltajes(t, v_a, v_b, nombre):
    graficar_columnas(
        x=[t, t],
        y=[v_a, v_b],
        tit=f"Voltaje vs Tiempo {nombre}",
        nom_x="Tiempo (s)",
        nom_y="Amplitud (V)",
        label=["Va", "Vb"]
    )

def graficar_corriente(v_b, i, nombre):
    graficar_columnas(
        x=[v_b],
        y=[i],
        tit=f"Corriente vs Voltaje {nombre}",
        nom_x="Voltaje (V)",
        nom_y="Corriente (A)",
        label=["I"],
        markersize=3,
        fmt="o"
    )

def graficar_regresion(x, y, std, ec_fit, nombre):
    graficar_columnas(
        x=x,
        y=y,
        tit=f"Corriente promedio vs Voltaje promedio {nombre}",
        nom_x="Voltaje (V)",
        nom_y="Corriente (A)",
        label=["I_promedio", f"Ajuste emergencial: {ec_fit[0]}", f"Ajuste no emergencial: {ec_fit[1]}"],
        markersize=[3, 0.1, 0.1],
        fmt=["o", "--", "--"],
        yerr=std
    )

def graficar_etas(temperaturas, etas, nombres_puntos):
    graficar_columnas(
        x=[temperaturas],
        y=[etas],
        tit="Factor de idealidad vs Longitud de onda",
        nom_x="Temperatura (°K)",
        nom_y="Factor de idealidad (n)",
        label=["n"],
        markersize=3,
        fmt="-o",
        names=[nombres_puntos]
    )

def graficar_regresiones(x, y, label):
    graficar_columnas(
        x=x,
        y=y,
        tit=label,
        nom_x="Voltaje (V)",
        nom_y="Corriente (A)",
        label=label
    )

def eta(a, c, q, k, temp, res, directorio_eta, offset, v_ventana):
    nombres_archivos_eta = filtrar_extension(directorio_eta)
    etas_dict = {}
    for nombre in nombres_archivos_eta:
        ruta_archivo = directorio_eta / nombre
        t, v_a, v_b, li = obtener_datos(ruta_archivo, offset, v_ventana[nombre])
        i = (v_a - v_b) / res
        filt_i, filt_v_b = filtrar_datos(i, v_b, li=li)
        v_umbral, i_promedio, i_std = valores_representativos(filt_v_b, filt_i)
        (v_fit_e, i_fit_e), b_e, = regresion_emergencial(v_umbral, i_promedio, a, c, i_std)
        (v_fit_ne, i_fit_ne), (a_ne, b_ne, c_ne), (_, b_std, _)= regresion_no_emergencial(v_umbral, i_promedio, a, b_e, c, i_std)
        ec_fit_e = f"I = {a:.2e} (exp({b_e:.2e} V) - 1) + {c:.2e}"
        ec_fit_ne = f"I = {a_ne:.2e} (exp({b_ne:.2e} V) - 1) + {c_ne:.2e}"
        eta = q / (k * temp * b_ne)
        color_led = nombre.split(".")[0]
        etas_dict.update({color_led: (eta, b_ne, b_std, ec_fit_ne)})
        graficar_voltajes(t, v_a, v_b, f"Rectificador a {color_led}")
        graficar_corriente(v_b, i, f"Rectificador a {color_led}")
        graficar_regresion(
            [v_umbral, v_fit_e, v_fit_ne],
            [i_promedio, i_fit_e, i_fit_ne],
            [i_std, None, None],
            [ec_fit_e, ec_fit_ne],
            f"Rectificador a {color_led}",
        )
    return etas_dict

def temperatura(a, c, q, k, eta, res, directorio_temp, offset, v_ventana):
    nombres_archivos_eta = filtrar_extension(directorio_temp)
    temp_dict = {}
    for nombre in nombres_archivos_eta:
        ruta_archivo = directorio_temp / nombre
        _, v_a, v_b, li = obtener_datos(ruta_archivo, offset, v_ventana[nombre])
        i = (v_a - v_b) / res
        filt_i, filt_v_b = filtrar_datos(i, v_b, li=li)
        v_umbral, i_promedio, i_std = valores_representativos(filt_v_b, filt_i)
        _, b_e, = regresion_emergencial(v_umbral, i_promedio, a, c, i_std)
        (v_fit_ne, i_fit_ne), (a_ne, b_ne, c_ne), (_, b_std, _)= regresion_no_emergencial(v_umbral, i_promedio, a, b_e, c, i_std)
        ec_fit_ne = f"I = {a_ne:.2e} (exp({b_ne:.2e} V) - 1) + {c_ne:.2e}"
        temp = q / (k * eta * b_ne)
        temp_c = temp - 273.15
        color_led = nombre.split(".")[0]
        temp_dict.update({color_led: ((temp, temp_c, b_ne, b_std, ec_fit_ne), (v_fit_ne, i_fit_ne, color_led))})
    return temp_dict

if __name__ == "__main__":
    a = 1e-9                # I_0
    c = 4.4e-4              # Componente dc
    q = 1.602176634e-19     # Carga elemental del electron
    k = 1.380649e-23        # Constante de Boltzmann
    temp = 298.15           # Temperatura en Kelvin
    res = 217.0             # Resistencia
    offset = 1.7            # Offset para evitar corriente negativa 
    voltajes_ventana = {    # Voltajes para filtrar datos
        "25°C": -0.7,
        "50°C": -0.7,
        "70°C": -0.7,
        "90°C": -0.7
    }
    temperaturas_kelvin = { 
        "25°C": 298.15,
        "50°C": 323.15,
        "70°C": 343.15,
        "90°C": 363.15
    }
    voltajes_lm_35 = {
        "25°C": 0.252,
        "50°C": 0.497,
        "70°C": 0.704,
        "90°C": 0.895
    }
    label_temp = "25°C"
    directorio_eta = Path("./datos_eta")
    directorio_temp = Path("./datos_temp")

    etas_dict = eta(a, c, q, k, temp, res, directorio_eta, offset, voltajes_ventana)
    temperatura_ambiente = etas_dict[label_temp][0]
    temps_dict = temperatura(a, c, q, k, temperatura_ambiente, res, directorio_temp, offset, voltajes_ventana)

    graficar_etas([temperaturas_kelvin[label_temp]], [etas_dict[label_temp][0]], [f"({label_temp}, {temperatura_ambiente:.4f})"])
    x, y, label = [], [], []
    for key in temps_dict.keys():
        x_, y_, label_ = temps_dict[key][1]
        x.append(x_)
        y.append(y_)
        label.append(label_)
    graficar_regresiones(x, y, label)
    with open("resultados.txt", "w", encoding="utf-8") as archivo:
        print("ETA DEL DIODO:".center(53, "="))
        for key in etas_dict.keys():
            n, b, std_b, ec = etas_dict[key]
            std_b = std_b * n / b
            text = f"Rectificador a {key.upper()}:\neta:\n\t{n}\nincertidumbre eta (+/-):\n\tb: {std_b}\necuacion:\n\t{ec}\n"
            print(text)
            archivo.write(text)
        print("TEMPERATURAS DEL DIODO:".center(53, "="))
        for key in temps_dict.keys():
            temp, temp_c, b, std_b, ec = temps_dict[key][0]
            std_b = abs(std_b * temp / b)
            text = f"Rectificador a {key.upper()}:\ntemperatura:\n\t{temp} °K\n\t{temp_c} °C\nincertidumbre temperatura (+/-):\n\tb: {std_b}\necuacion:\n\t{ec}\n"
            print(text)
            archivo.write(text)