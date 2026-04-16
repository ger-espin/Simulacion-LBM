# LBM D2Q9 BGK - flujo 2D alrededor de un cilindro fijo

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import time
import os

# ==================== CONSTANTES Y PARÁMETROS DEL MODELO ======================

# Esquema D2Q9
C = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
])
W = np.array([4/9] + [1/9] * 4 + [1/36] * 4) # Pesos
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6]) # Direcciones opuestas

# --- Parámetros ---
D = 30.0            # Diámetro del cilindro
U_INLET = 0.04      # Velocidad de entrada (unidades lattice)
RE_TARGET = 210.0   # Número de Reynolds 

# --- Parámetros de Simulación y Geometría ---
NX = 400            # Tamaño X
NY = 200            # Tamaño Y
CX = NX // 5        # Posición X del centro del cilindro
CY = NY // 2        # Posición Y del centro del cilindro
MAX_ITERS = 200000
OUTPUT_INTERVAL = 500
OUTPUT_DIR = 'img_sequence'

# --- Cálculo de Parámetros LBM ---
CS2 = 1.0 / 3.0
NU = U_INLET * D / RE_TARGET # Viscosidad cinemática (nu)
TAU = 0.5 + NU / CS2         # Tiempo de relajación
OMEGA = 1.0 / TAU            # Frecuencia de colisión

# Verificar estabilidad
if TAU < 0.5:
    raise ValueError(f"TAU ({TAU:.4f}) debe ser >= 0.5 para estabilidad.")

class LBM_Simulator:
    def __init__(self, nx, ny, cx, cy, d, u_inlet, omega):
        # 1. Parámetros y Geometría
        self.nx, self.ny = nx, ny
        self.omega = omega
        self.u_inlet = u_inlet
        
        # 2. Inicialización de arrays
        self.f = np.zeros((9, ny, nx)) # Distribuciones f_k
        self.rho = np.ones((ny, nx))
        self.ux = np.zeros((ny, nx))
        self.uy = np.zeros((ny, nx))
        
        # 3. Máscara del cilindro
        Y, X = np.indices((ny, nx))
        self.mask = (X - cx)**2 + (Y - cy)**2 <= (d / 2.0)**2
        
        # 4. Inicializar poblaciones f a su valor de equilibrio
        for k in range(9):
            self.f[k, :, :] = self._feq(k, self.rho, self.ux, self.uy)

    def _feq(self, k, rho, ux, uy):
        """Calcula la distribución de equilibrio feq_k."""
        cu = 3.0 * (C[k, 0] * ux + C[k, 1] * uy)
        uu = 1.5 * (ux**2 + uy**2)
        return W[k] * rho * (1.0 + cu + 0.5 * cu**2 - uu)

    def calcular_macro(self):
        """Calcula densidad (rho) y velocidad (ux, uy) a partir de f."""
        self.rho = np.sum(self.f, axis=0)
        self.ux = np.sum(self.f * C[:, 0, None, None], axis=0) / self.rho
        self.uy = np.sum(self.f * C[:, 1, None, None], axis=0) / self.rho

    def aplicar_condiciones_frontera(self):
        """Aplica la condición de velocidad constante en la entrada (x=0) 
        y la condición simple de salida (x=Nx-1)."""
        
        # Entrada : Perfil de velocidad fijo
        # Simplificación: Asume que las poblaciones f[k,:,0] se ajustan 
        # para dar la velocidad Ux=U_INLET.
        self.ux[:, 0] = self.u_inlet
        self.uy[:, 0] = 0.0
        self.rho[:, 0] = 1.0 # Densidad constante a la entrada
        
        # Reconstruir las poblaciones 'entrantes' f_k en x=0
        # (k=1, 5, 8 son las que apuntan hacia el interior desde x=0)
        for k in [1, 5, 8]:
             self.f[k, :, 0] = self._feq(k, self.rho[:, 0], self.ux[:, 0], self.uy[:, 0])

        self.f[:, :, -1] = self.f[:, :, -2]

    def colision_bgk(self):
        """Paso de Colisión usando el modelo BGK."""
        feq_all = np.zeros_like(self.f)
        for k in range(9):
            feq_all[k, :, :] = self._feq(k, self.rho, self.ux, self.uy)
            
        # Colisión BGK: f_new = f_old + omega * (f_eq - f_old)
        self.f += self.omega * (feq_all - self.f)

    def streaming(self):
        """Paso de Streaming (propagación) usando roll de numpy."""
        for k in range(9):
            self.f[k] = np.roll(np.roll(self.f[k], C[k, 1], axis=0), C[k, 0], axis=1)

    def bounce_back_y_fuerza(self):
        """Implementa la condición de frontera 'Bounce-Back' sobre el obstáculo 
        y calcula el intercambio de momento (fuerza)."""
        
        Fx_loc, Fy_loc = 0.0, 0.0
        
        solid_indices = np.argwhere(self.mask)
        
        for (iy, ix) in solid_indices:
            for k in range(9):
                iyn = iy - C[k, 1]
                ixn = ix - C[k, 0]
                
                if (0 <= iyn < self.ny) and (0 <= ixn < self.nx):
                    if not self.mask[iyn, ixn]:
                        f_coming = self.f[k, iyn, ixn]
                        
                        # B) Intercambio de Momento (Force): 2 * f_coming * c_k
                        # El 2 es porque se refleja.
                        Fx_loc += 2.0 * f_coming * C[k, 0]
                        Fy_loc += 2.0 * f_coming * C[k, 1]
                        
                        # C) Bounce-Back: la población f_k que llega rebota como f_opp[k] hacia atrás
                        # Escribir en la posición (iyn, ixn) la población opuesta.
                        # NOTA: La población entrante f_k que iba a (iy,ix) nunca llega allí 
                        # en el streaming global; solo se usa para calcular Fx_loc, Fy_loc.
                        # El bounce-back se implementa "escribiendo" la distribución opuesta 
                        # en el nodo fluido (iyn, ixn).
                        self.f[OPP[k], iyn, ixn] = f_coming
                        
        # La fuerza sobre el cilindro es el negativo de Fx_loc, Fy_loc
        return -Fx_loc, -Fy_loc

#---------------- POST-PROCESADO Y VISUALIZACIÓN ----------------------


def post_process_results(t_arr, Fx_time, Fy_time, u_ref, d_ref, re_target, 
                         output_dir, speed_final, nx, ny):
    """Calcula coeficientes y genera gráficas de salida."""
    
    # Parámetros de referencia
    rho0 = 1.0 # Densidad de referencia
    
    # Cd = Fx / (0.5 * rho * U^2 * D)
    Cd = np.array(Fx_time) / (0.5 * rho0 * u_ref**2 * d_ref)
    Cl = np.array(Fy_time) / (0.5 * rho0 * u_ref**2 * d_ref)
    
    # --- Cálculo de Strouhal (St) por FFT ---
    Nfft = len(Cl)
    Cl_stable = Cl[int(Nfft * 0.3):]
    t_stable = t_arr[int(Nfft * 0.3):]

    # Eliminar media para enfocarse en la oscilación
    cl_detrended = Cl_stable - np.mean(Cl_stable) 
    
    yf = np.abs(rfft(cl_detrended))
    xf = rfftfreq(len(Cl_stable), d=1.0) # Frecuencia en ciclos por iteración
    
    # Encontrar la frecuencia pico (excluir la primera componente DC)
    if len(yf[1:]) > 0:
        idx_peak = np.argmax(yf[1:]) + 1
        f_peak = xf[idx_peak]
        St_est = f_peak * d_ref / u_ref
    else:
        f_peak = 0.0
        St_est = 0.0

    # --- Salida ---
    print("\n--- Resultados de Simulación ---")
    print(f"Re (target) = {re_target:.1f} ; St (estimado) = {St_est:.4f}")
    print(f"Cd promedio final (última {len(Cl_stable)} it) = {np.mean(Cd[-len(Cl_stable):]):.4f}")
    
    # --- Guardar Datos y Gráficas ---
    
    # 1. Guardar datos CSV
    data_out = np.stack((t_arr, Cd, Cl), axis=1)
    np.savetxt(f"resultados_Re{re_target:.0f}.csv", data_out, 
               header="Iter, Cd, Cl", delimiter=",", comments="")
    
    # 2. Gráficas
    plt.ioff() # Desactivar modo interactivo para generación de archivos
    
    # Gráfica 1: Series temporales
    plt.figure(figsize=(10, 5))
    plt.plot(t_arr, Cd, label='Cd(t)')
    plt.plot(t_arr, Cl, label='Cl(t)')
    plt.legend()
    plt.xlabel('Iteración')
    plt.ylabel('Coeficiente')
    plt.title(f'Series temporales Cd y Cl (Re={re_target:.0f})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"TimeSeries_Re{re_target:.0f}.png"))
    plt.close()

    # Gráfica 2: Espectro FFT
    plt.figure(figsize=(10, 5))
    plt.plot(xf, yf)
    if f_peak > 0:
        plt.axvline(f_peak, color='r', linestyle='--', label=f'Peak f={f_peak:.4e}')
    plt.xlabel('Frecuencia (1/iteración)')
    plt.ylabel('Amplitud FFT')
    plt.legend()
    plt.title(f'Espectro de Cl (St estimado={St_est:.4f})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"FFT_Cl_Re{re_target:.0f}.png"))
    plt.close()

    # Gráfica 3: Campo de velocidad final
    plt.figure(figsize=(nx/50, ny/50))
    plt.imshow(speed_final, origin='lower', cmap='viridis')
    plt.colorbar(label='|u|')
    plt.title(f'Campo de Velocidad Final (Re={re_target:.0f})')
    # Añadir el cilindro para visualizar
    circle = plt.Circle((CX, CY), D/2.0, color='red', fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.savefig(os.path.join(output_dir, f"VelocidadFinal_Re{re_target:.0f}.png"))
    plt.close()

def main():
  
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    sim = LBM_Simulator(NX, NY, CX, CY, D, U_INLET, OMEGA)
    
    # Arreglos para almacenamiento de resultados
    Fx_time, Fy_time, time_series = [], [], []
    
    # Inicializar figura de visualización
    plt.ioff() 
    fig, ax = plt.subplots(figsize=(NX/50, NY/50))
    
    # 2. Bucle principal
    t0 = time.time()
    for it in range(MAX_ITERS):
        # A) Cálculo de macroscópicos, Colisión y Streaming
        sim.calcular_macro() 
        sim.colision_bgk()
        sim.streaming()
        
        # B) Condiciones de Frontera
        sim.aplicar_condiciones_frontera()
        
        # C) Bounce-Back y Cálculo de Fuerza
        Fx, Fy = sim.bounce_back_y_fuerza()
        
        # D) Almacenamiento y Visualización
        Fx_time.append(Fx)
        Fy_time.append(Fy)
        time_series.append(it)
        
        if (it % OUTPUT_INTERVAL == 0) or (it == MAX_ITERS - 1):
            speed = np.sqrt(sim.ux**2 + sim.uy**2)
            
            # 1. Visualización interactiva
            ax.clear()
            im = ax.imshow(speed, origin='lower', cmap='viridis')
            ax.set_title(f'Campo de Velocidad (Re={RE_TARGET:.0f}) - Iter {it}')
            
            # Dibuja el cilindro también
            circle = plt.Circle((CX, CY), D/2.0, color='red', fill=False, linewidth=1)
            ax.add_patch(circle)
            
            # Añadir barra de color
            if it == 0: fig.colorbar(im, ax=ax, label='|u|')
            
            #plt.draw()
            #plt.pause(0.001)

            # 2. Guardar imagen
            filename = os.path.join(OUTPUT_DIR, "vel.{0:05d}.png".format(it // OUTPUT_INTERVAL))
            fig.savefig(filename)
            print(f"Iter {it}: Imagen guardada y Fx={Fx:.6f}")


    t_elapsed = time.time() - t0
    print(f"\nTiempo total de simulación (s): {t_elapsed:.2f}")

    # 3. Post-Procesado y Guardado Final
    speed_final = np.sqrt(sim.ux**2 + sim.uy**2)
    post_process_results(
        time_series, Fx_time, Fy_time, U_INLET, D, RE_TARGET, OUTPUT_DIR, 
        speed_final, NX, NY
    )
    plt.ioff() # Volver a modo no interactivo
    #plt.show()

if __name__ == "__main__":
    main()
