# Helmet Vision Dashboard

Frontend moderno en React + TailwindCSS para un sistema de detección de uso de casco en imágenes estáticas.

## Incluye
- Sidebar colapsable
- Carga de imágenes estáticas
- Vista principal con bounding boxes simulados
- Historial de detecciones recientes
- Panel inferior reservado para estadísticas futuras
- Estructura preparada para reemplazar la simulación por una API real

## Tecnologías
- React con hooks
- Vite
- TailwindCSS
- JSX

## Estructura
- `src/components/Sidebar.jsx`
- `src/components/ImageUploader.jsx`
- `src/components/DetectionViewer.jsx`
- `src/components/DetectionList.jsx`
- `src/components/StatsPanel.jsx`
- `src/data/mockDetections.js`
- `src/services/mockDetectionService.js`

## Ejecutar
1. Entrar en la carpeta `frontend`
2. Instalar dependencias:

```bash
npm install
```

3. Levantar el entorno de desarrollo:

```bash
npm run dev
```

## Notas
- No usa cámara en vivo.
- Todo está basado en imágenes estáticas.
- La lógica de `mockDetectionService` se puede reemplazar en el futuro por una llamada a tu backend de inferencia.
