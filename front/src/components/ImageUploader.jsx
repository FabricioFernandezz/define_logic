import { useState } from "react";

const formatFileSize = (size) => {
  if (!size) {
    return "0 KB";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
};

export default function ImageUploader({ image, onSelectImage, onClearImage }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleChange = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      onSelectImage(file);
    }
    event.target.value = "";
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      onSelectImage(file);
    }
  };

  return (
    <section id="upload-section" className="rounded-[2rem] border border-white/8 bg-white/5 p-5 shadow-glow backdrop-blur-xl">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-accent-300/75">Carga</p>
          <h2 className="mt-1 text-2xl font-semibold text-white">Cargar imágenes</h2>
          <p className="mt-2 text-sm leading-6 text-steel-300">
            Arrastra una imagen o selecciónala para visualizar el resultado del detector.
          </p>
        </div>

        <label className="inline-flex cursor-pointer items-center justify-center rounded-2xl bg-gradient-to-r from-accent-500 to-ok-500 px-5 py-3 text-sm font-semibold text-steel-950 transition hover:brightness-110">
          Seleccionar archivo
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleChange}
          />
        </label>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`rounded-[1.75rem] border border-dashed bg-steel-950/70 p-4 transition ${
            isDragging ? "border-accent-400/60 bg-accent-500/5" : "border-white/10"
          }`}
        >
          {image ? (
            <div className="space-y-4">
              <div className="overflow-hidden rounded-[1.5rem] border border-white/8 bg-steel-900/90">
                <img
                  src={image.previewUrl}
                  alt={image.name}
                  className="h-64 w-full object-cover sm:h-72"
                />
              </div>

              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm font-medium text-white">{image.name}</p>
                  <p className="text-xs text-steel-400">
                    {image.naturalWidth} × {image.naturalHeight} px
                  </p>
                </div>
                <div className="text-xs text-steel-400">
                  Tamaño: {formatFileSize(image.file?.size)}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex h-full min-h-[280px] flex-col items-center justify-center rounded-[1.5rem] bg-gradient-to-br from-white/5 to-transparent px-6 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-[1.5rem] bg-accent-500/15 text-3xl text-accent-200">
                ⬆
              </div>
              <p className="mt-4 text-lg font-medium text-white">
                {isDragging ? "Suelta la imagen aquí" : "Arrastra una imagen o selecciónala"}
              </p>
              <p className="mt-2 max-w-md text-sm leading-6 text-steel-400">
                El frontend está preparado para trabajar con imágenes de obra y mostrar el resultado del modelo sobre el visor principal.
              </p>
            </div>
          )}
        </div>

        <div className="flex flex-col justify-between gap-4 rounded-[1.75rem] border border-white/8 bg-steel-900/70 p-4">
          <div>
            <p className="text-sm font-medium text-white">Flujo de trabajo</p>
            <ul className="mt-3 space-y-3 text-sm text-steel-300">
              <li className="flex gap-3">
                <span className="mt-1 h-2.5 w-2.5 rounded-full bg-accent-400" />
                Subir imagen estática desde el disco.
              </li>
              <li className="flex gap-3">
                <span className="mt-1 h-2.5 w-2.5 rounded-full bg-ok-400" />
                Previsualizar antes de procesar.
              </li>
              <li className="flex gap-3">
                <span className="mt-1 h-2.5 w-2.5 rounded-full bg-warn-400" />
                Mostrar boxes y etiquetas simuladas.
              </li>
            </ul>
          </div>

          <button
            type="button"
            onClick={onClearImage}
            disabled={!image}
            className="inline-flex items-center justify-center rounded-2xl border border-white/8 bg-white/5 px-4 py-3 text-sm font-medium text-white transition hover:border-white/15 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Limpiar imagen cargada
          </button>
        </div>
      </div>
    </section>
  );
}