# ClasificameAI-Espanol
Es clasificador simple para categorizar archivos txt o markdown (md) por carpetas en base al contenido analizado por AI usando la GPU

## Que hace el script?

Analiza una carpeta con archivos de texto, lee el texto y en base al texto determina que categoria o tematica es mas probable que corresponda ese text (Las categorias o tematicas son indicados por usted, entonces usted debe especificar cuales desea usar para que la AI las clasifique), luego el script copiara sus archivos a una carpeta de salida organados por nuevas carpetas segun las categorias que usted indico y que la inteligencia artifical considero mas aptas.

### Requerimientos

GPU NVIDIA
```pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.2 tqdm protobuf==3.20.0 sentencepiece accelerate tiktoken
```

o sino usa mi archivo requirements por si olvide poner algo.

```pip
pip install -r requirements.txt
```

## Forma de uso

Necesitas un folder que contenga todas las notas o archivos del texto de formato `.md` o `.txt` y una carpeta de salida, donde el script copiara todos los archivos que analice y considere aptos para esa categoria o tema

```python
python clasificador_espanol.py --input "./notas" --output "./salida" --categories "Juegos" "Personal" "Finanzas" "Impuestos" "Salud"
```

## Notas 
El script necesita internet para descargar los modelos de AI de hugging face, lo deberia hacer automaticamente.

Dentro del script hay una linea que dice:

```python
max_length = 2048
```
Puedes reducir el numero de `2048` de la variable `max_length` si tienes una GPU con poca memoria, por ejemplo en 512.
