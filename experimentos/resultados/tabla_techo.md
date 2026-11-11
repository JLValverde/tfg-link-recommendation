# Techo del recuperador (recall@K)

Recall@K del recuperador (Qwen/Qwen3-Embedding-0.6B): fracción de los enlaces de referencia de cada consulta que entra entre los K primeros, media [IC 95 %] sobre las 1800 consultas. Los sistemas puntúan las K = 50 primeras, así que recall@50 es el máximo que puede recuperar cualquiera. La última columna es el % de consultas sin ningún enlace de referencia entre esas 50. Generada por `experimentos/tablas_finales.py`.

| Escenario | Tarea | recall@10 | recall@20 | **recall@50** | recall@100 | Listas sin enlace de referencia (K = 50) |
|---|---|---|---|---|---|---|
| frío | completa | 0,173 [0,166, 0,180] | 0,225 [0,217, 0,234] | 0,310 [0,300, 0,319] | 0,385 [0,374, 0,395] | 9,2 % (166) |
| frío | implícita | 0,128 [0,120, 0,135] | 0,173 [0,164, 0,182] | 0,250 [0,240, 0,261] | 0,325 [0,314, 0,337] | 22,6 % (407) |
| frío | implícita solo ancla | 0,129 [0,122, 0,137] | 0,175 [0,166, 0,184] | 0,253 [0,243, 0,263] | 0,328 [0,317, 0,339] | 21,8 % (392) |
| caliente | completa | 0,169 [0,160, 0,177] | 0,219 [0,209, 0,229] | 0,305 [0,294, 0,316] | 0,377 [0,363, 0,388] | 20,5 % (369) |
| caliente | implícita | 0,133 [0,123, 0,142] | 0,178 [0,168, 0,188] | 0,260 [0,246, 0,272] | 0,331 [0,317, 0,345] | 35,4 % (638) |
| caliente | implícita solo ancla | 0,134 [0,125, 0,143] | 0,179 [0,169, 0,190] | 0,262 [0,249, 0,274] | 0,334 [0,320, 0,347] | 34,6 % (623) |
