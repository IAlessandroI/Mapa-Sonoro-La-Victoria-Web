"""Página web: sube un audio o graba con el micrófono y detecta el tipo de sonido."""
import gradio as gr
from predictor import ModeloSonidos
from caracteristicas import extraer

modelo = ModeloSonidos()
NOMBRES = {"claxons": "Claxon (bocina)", "fondo": "Ruido de fondo",
           "ladridos": "Ladrido", "risas": "Risas"}

def clasificar(ruta):
    if not ruta:
        return None, "Sube un audio o graba uno con el micrófono."
    try:
        f = extraer(ruta)
    except Exception as e:
        return None, f"No se pudo leer el audio: {e}"
    clase, probs = modelo.predecir([f[c] for c in modelo.columnas])
    resultado = {NOMBRES[c]: p for c, p in probs.items()}
    texto = f"### Sonido detectado: {NOMBRES[clase]}"
    if max(probs.values()) < 0.5:
        texto += "\n\nEl modelo no está muy seguro. Prueba con un audio más claro y de un solo sonido."
    return resultado, texto

demo = gr.Interface(
    fn=clasificar,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio"),
    outputs=[gr.Label(num_top_classes=4, label="Probabilidades"), gr.Markdown()],
    title="Mapa Sonoro: detector de sonidos de la ciudad",
    description=("Sube un audio o graba con el micrófono (de 3 a 10 segundos, con un solo sonido). "
                 "El modelo (SVM) reconoce: claxons, ladridos, risas y ruido de fondo."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
