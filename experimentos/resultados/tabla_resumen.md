# Tabla resumen (modo recuperador, tarea implícita)

Media [IC 95 %] sobre las 1800 consultas de la muestra principal que no son de desarrollo, y latencia p50 por consulta (en los sistemas con modelo de lenguaje, sin las 3 consultas de calentamiento y sobre las 1800; en los demás, lo medido: las 2000 menos las 3 de calentamiento). Generada por `experimentos/tablas_finales.py`.

VRAM: nivel de presupuesto (tope) y, en GB, el pico de la propia ejecución (el mayor de sus escenarios) / el pico de T2.6 con el modelo de embeddings cargado a la vez, que es el que se comprueba contra el tope. Los sistemas sin modelo de lenguaje no tienen nivel. Las tres tareas y todas las diferencias, en `tabla_principal.md`.

| Sistema | VRAM: nivel (tope): pico / pico T2.6 | frío: precision@1 | frío: precision_norm@5 | frío: MRR | frío: hits@5 | frío: latencia p50 | caliente: precision@1 | caliente: precision_norm@5 | caliente: MRR | caliente: hits@5 | caliente: latencia p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aleatorio (control) | — | 0,081 [0,069, 0,093] | 0,078 [0,071, 0,085] | 0,183 [0,170, 0,195] | 0,271 [0,251, 0,289] | — | 0,040 [0,031, 0,049] | 0,053 [0,047, 0,059] | 0,116 [0,106, 0,126] | 0,176 [0,159, 0,194] | — |
| oráculo (control) | — | 0,774 [0,755, 0,791] | 0,445 [0,428, 0,461] | 0,774 [0,755, 0,791] | 0,774 [0,755, 0,791] | — | 0,646 [0,624, 0,667] | 0,335 [0,321, 0,350] | 0,646 [0,624, 0,667] | 0,646 [0,624, 0,667] | — |
| C0 | — | 0,339 [0,317, 0,363] | 0,215 [0,204, 0,227] | 0,435 [0,415, 0,455] | 0,543 [0,520, 0,565] | 29,9 ms | 0,213 [0,194, 0,231] | 0,149 [0,140, 0,160] | 0,299 [0,281, 0,316] | 0,388 [0,366, 0,409] | 0,323 ms |
| BM25 | — | 0,339 [0,317, 0,361] | 0,219 [0,208, 0,230] | 0,441 [0,422, 0,460] | 0,568 [0,546, 0,589] | 0,387 ms | 0,211 [0,191, 0,230] | 0,155 [0,145, 0,165] | 0,304 [0,286, 0,321] | 0,416 [0,392, 0,437] | 0,339 ms |
| vecinos comunes (B) | — | no aplica | no aplica | no aplica | no aplica | — | 0,287 [0,266, 0,308] | 0,193 [0,181, 0,204] | 0,372 [0,352, 0,391] | 0,473 [0,448, 0,496] | 0,040 ms |
| Adamic-Adar (B) | — | no aplica | no aplica | no aplica | no aplica | — | 0,311 [0,289, 0,331] | 0,206 [0,194, 0,217] | 0,394 [0,373, 0,414] | 0,496 [0,472, 0,519] | 0,058 ms |
| fusión RRF BM25 + Adamic-Adar (B) | — | no aplica | no aplica | no aplica | no aplica | — | 0,279 [0,260, 0,299] | 0,194 [0,182, 0,205] | 0,372 [0,353, 0,392] | 0,482 [0,459, 0,504] | 0,793 ms |
| C1 (qwen2.5:3b) | bajo (8,0 GB): 4,1 / 7,5 GB | 0,345 [0,323, 0,366] | 0,237 [0,225, 0,249] | 0,459 [0,441, 0,477] | 0,606 [0,584, 0,627] | 2,37 s | 0,213 [0,194, 0,231] | 0,169 [0,157, 0,179] | 0,314 [0,295, 0,331] | 0,439 [0,417, 0,460] | 2,33 s |
| C1 (qwen2.5:7b) | medio (12,0 GB): 6,0 / 10,0 GB | 0,329 [0,309, 0,350] | 0,244 [0,232, 0,255] | 0,455 [0,438, 0,474] | 0,622 [0,600, 0,644] | 3,47 s | 0,204 [0,188, 0,222] | 0,177 [0,166, 0,188] | 0,316 [0,300, 0,334] | 0,455 [0,433, 0,477] | 3,40 s |
| C1 (qwen2.5:14b) | alto (16,0 GB): 10,8 / 14,6 GB | 0,390 [0,367, 0,412] | 0,272 [0,259, 0,284] | 0,504 [0,484, 0,523] | 0,650 [0,628, 0,672] | 5,94 s | 0,257 [0,237, 0,276] | 0,198 [0,187, 0,209] | 0,361 [0,342, 0,379] | 0,494 [0,473, 0,516] | 5,91 s |
| C2 (qwen2.5:3b) | bajo (8,0 GB): 3,4 / 7,5 GB | 0,333 [0,311, 0,354] | 0,233 [0,221, 0,245] | 0,452 [0,433, 0,471] | 0,602 [0,581, 0,625] | 2,72 s | 0,186 [0,168, 0,202] | 0,163 [0,153, 0,173] | 0,298 [0,281, 0,314] | 0,438 [0,415, 0,459] | 2,59 s |
| C2 (qwen2.5:7b) | medio (12,0 GB): 6,9 / 10,0 GB | 0,328 [0,306, 0,349] | 0,240 [0,228, 0,251] | 0,449 [0,430, 0,467] | 0,612 [0,590, 0,633] | 4,08 s | 0,174 [0,156, 0,192] | 0,166 [0,155, 0,176] | 0,291 [0,274, 0,307] | 0,434 [0,411, 0,456] | 4,00 s |
| C2 (qwen2.5:14b) | alto (16,0 GB): 10,8 / 14,6 GB | 0,383 [0,361, 0,404] | 0,269 [0,257, 0,282] | 0,497 [0,477, 0,515] | 0,644 [0,623, 0,667] | 7,22 s | 0,218 [0,199, 0,236] | 0,193 [0,182, 0,205] | 0,335 [0,318, 0,352] | 0,487 [0,465, 0,509] | 7,12 s |

## Análisis exploratorio (a posteriori)

**No son sistemas del protocolo.** Se añaden en T2.8b, por decisión de Jose, después de ver los resultados de C1 y de los baselines. Fusión RRF (k = 60) de C1 con cada modelo y Adamic-Adar (B), como la de BM25 y Adamic-Adar (B): rangos del arnés, desempate fijo y lista entera. Se calcula desde las puntuaciones guardadas de C1, sin volver a llamar al modelo (`experimentos/evaluar_fusion_exploratoria.py`). Solo caliente y modo recuperador. Sin latencia propia: no se ha medido (su coste es el de C1 con ese modelo más el de Adamic-Adar (B)).

| Sistema | caliente: precision@1 | caliente: precision_norm@5 | caliente: MRR | caliente: hits@5 |
|---|---|---|---|---|
| fusión RRF C1 (qwen2.5:3b) + Adamic-Adar (B) | 0,292 [0,270, 0,313] | 0,206 [0,193, 0,217] | 0,388 [0,369, 0,407] | 0,503 [0,478, 0,526] |
| fusión RRF C1 (qwen2.5:7b) + Adamic-Adar (B) | 0,298 [0,277, 0,319] | 0,215 [0,203, 0,226] | 0,396 [0,376, 0,414] | 0,522 [0,499, 0,544] |
| fusión RRF C1 (qwen2.5:14b) + Adamic-Adar (B) | 0,321 [0,300, 0,342] | 0,220 [0,209, 0,232] | 0,413 [0,394, 0,432] | 0,533 [0,510, 0,555] |
