# Ablación: diferencias pareadas (modo recuperador, tarea implícita)

Diferencia media A − B por consulta [IC 95 % por bootstrap pareado, 1000 remuestreos], sobre las 1800 consultas. Con * las que no incluyen el 0. Son las de `fig_8_ablacion`, con sus bloques; las tres tareas y las cuatro métricas, en `tabla_principal.md`. Generada por `experimentos/tablas_finales.py`.

| Bloque | Comparación | Escenario | MRR | precision@1 |
|---|---|---|---|---|
| Efecto de la reordenación | C1 − C0 (3B) | frío | +0,024 [+0,006, +0,042] * | +0,006 [−0,018, +0,030] |
| Efecto de la reordenación | C1 − C0 (3B) | caliente | +0,015 [−0,000, +0,030] | +0,000 [−0,019, +0,020] |
| Efecto del contexto | C2 − C1 (3B) | frío | −0,007 [−0,016, +0,002] | −0,012 [−0,028, +0,002] |
| Efecto del contexto | C2 − C1 (3B) | caliente | −0,016 [−0,027, −0,006] * | −0,027 [−0,044, −0,011] * |
| Efecto del contexto | C2 − C1 (7B) | frío | −0,006 [−0,015, +0,002] | −0,001 [−0,016, +0,013] |
| Efecto del contexto | C2 − C1 (7B) | caliente | −0,026 [−0,035, −0,016] * | −0,031 [−0,046, −0,016] * |
| Efecto del contexto | C2 − C1 (14B) | frío | −0,007 [−0,015, +0,001] | −0,007 [−0,021, +0,007] |
| Efecto del contexto | C2 − C1 (14B) | caliente | −0,026 [−0,036, −0,016] * | −0,039 [−0,056, −0,024] * |
| Efecto del tamaño | C1 (7B) − C1 (3B) | frío | −0,003 [−0,016, +0,008] | −0,016 [−0,034, +0,002] |
| Efecto del tamaño | C1 (7B) − C1 (3B) | caliente | +0,003 [−0,011, +0,014] | −0,008 [−0,027, +0,009] |
| Efecto del tamaño | C1 (14B) − C1 (7B) | frío | +0,048 [+0,037, +0,061] * | +0,061 [+0,041, +0,079] * |
| Efecto del tamaño | C1 (14B) − C1 (7B) | caliente | +0,045 [+0,034, +0,056] * | +0,053 [+0,035, +0,069] * |
| Efecto del tamaño | C1 (14B) − C1 (3B) | frío | +0,045 [+0,032, +0,059] * | +0,045 [+0,025, +0,065] * |
| Efecto del tamaño | C1 (14B) − C1 (3B) | caliente | +0,047 [+0,035, +0,059] * | +0,044 [+0,026, +0,063] * |

El análisis secundario (solo las consultas con algún enlace de referencia entre las candidatas) está en `tabla_ablacion_subconjunto.md`, siempre junto a estas cifras.
