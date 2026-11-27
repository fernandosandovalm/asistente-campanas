# Glosario de métricas — Social (Ben & Frank)

## Métricas base (columnas del data lake)
- **media_cost**: inversión en medios en USD reportada por la plataforma para ese día, país y creativo.
- **impressions**: número de veces que el anuncio se mostró en pantalla.
- **clicks**: clicks en el anuncio que llevan al sitio de Ben & Frank.
- **lpv (Landing Page Views)**: veces que la página de destino terminó de cargar después de un click. Siempre es menor o igual a los clicks; la diferencia son usuarios que abandonan antes de que cargue la página.
- **subscriptions**: citas de examen visual agendadas en el sitio de Ben & Frank, atribuidas al anuncio. Es la conversión principal de la campaña.
- **video_starts**: reproducciones iniciadas de un creativo de video. Solo aplica a creativos de video; en estáticos es 0.
- **video_completes**: reproducciones vistas hasta el final (100 %).

## KPIs calculados
- **CPM (costo por mil impresiones)** = media_cost / impressions × 1000. Mide qué tan caro es comprar alcance.
- **CPC (costo por click)** = media_cost / clicks.
- **CTR (click-through rate)** = clicks / impressions × 100. Mide qué tan atractivo es el anuncio para generar acción.
- **Tasa LPV** = lpv / clicks × 100. Calidad del click: si es muy baja (menos de 30 %), sospechar clicks accidentales o tráfico inválido (bots).
- **Costo por LPV** = media_cost / lpv.
- **Tasa de conversión** = subscriptions / lpv × 100. Porcentaje de visitas a la landing que agendan un examen visual.
- **Costo por cita (CPA)** = media_cost / subscriptions. Es el KPI de eficiencia principal para funnel lower.
- **View rate** = video_starts / impressions × 100.
- **VTR (tasa de completado)** = video_completes / video_starts × 100. Disminuye con la duración del video: un video de 6 s se completa mucho más que uno de 30 s.
- **CPV (costo por reproducción)** = media_cost / video_starts.
- **CPCV (costo por video completo)** = media_cost / video_completes.

## Reglas de interpretación
- Las tasas se calculan siempre sobre los totales agregados (suma de clicks / suma de impresiones), nunca promediando tasas diarias.
- Comparar CTR o CPA entre funnels distintos es engañoso: upper busca alcance y lower busca conversión.
- Con menos de 10.000 impresiones los KPIs son muy volátiles; mencionarlo antes de sacar conclusiones.
