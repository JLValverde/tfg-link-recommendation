# Salidas malformadas y casos del sí y el no (modo recuperador)

Sobre las 1800 consultas. Una llamada por candidato. Casos de la confianza P-cota: (i) "yes" y "no" están entre los 20 primeros *tokens*; (ii) falta uno y se acota con el menor del top; (iii) faltan los dos (puntuación None). None es la peor puntuación posible. Generada por `experimentos/tablas_finales.py`.

| Configuración | Modelo | Escenario | Llamadas | Errores | (i) ambos | (ii) falta uno | (iii) faltan ambos | None (tasa) | Respuestas que no son yes ni no | Truncados | Respuestas "Yes" |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | qwen2.5:3b | frío | 80 790 | 0 | 80 790 | 0 (0,0 %) | 0 | 0 (0,00 %) | 0 (0,0000 %) | 0 | 6,58 % |
| C1 | qwen2.5:3b | caliente | 80 395 | 0 | 80 394 | 1 (0,0 %) | 0 | 0 (0,00 %) | 1 (0,0012 %) | 0 | 5,33 % |
| C1 | qwen2.5:7b | frío | 80 790 | 0 | 66 147 | 14 643 (18,1 %) | 0 | 0 (0,00 %) | 0 (0,0000 %) | 0 | 17,97 % |
| C1 | qwen2.5:7b | caliente | 80 395 | 0 | 65 108 | 15 287 (19,0 %) | 0 | 0 (0,00 %) | 0 (0,0000 %) | 0 | 15,84 % |
| C1 | qwen2.5:14b | frío | 80 790 | 1 | 45 562 | 35 227 (43,6 %) | 0 | 1 (0,00 %) | 0 (0,0000 %) | 0 | 13,33 % |
| C1 | qwen2.5:14b | caliente | 80 395 | 0 | 43 651 | 36 744 (45,7 %) | 0 | 0 (0,00 %) | 0 (0,0000 %) | 0 | 10,90 % |
| C2 | qwen2.5:3b | frío | 80 790 | 0 | 80 790 | 0 (0,0 %) | 0 | 0 (0,00 %) | 0 (0,0000 %) | 0 | 9,93 % |
| C2 | qwen2.5:3b | caliente | 80 395 | 0 | 80 395 | 0 (0,0 %) | 0 | 0 (0,00 %) | 1 (0,0012 %) | 0 | 13,24 % |
| C2 | qwen2.5:7b | frío | 80 790 | 0 | 65 084 | 15 706 (19,4 %) | 0 | 0 (0,00 %) | 0 (0,0000 %) | 0 | 17,07 % |
| C2 | qwen2.5:7b | caliente | 80 395 | 0 | 67 724 | 12 671 (15,8 %) | 0 | 0 (0,00 %) | 0 (0,0000 %) | 0 | 18,23 % |
| C2 | qwen2.5:14b | frío | 80 790 | 0 | 48 526 | 32 264 (39,9 %) | 0 | 0 (0,00 %) | 0 (0,0000 %) | 0 | 14,65 % |
| C2 | qwen2.5:14b | caliente | 80 395 | 0 | 57 758 | 22 637 (28,2 %) | 0 | 0 (0,00 %) | 0 (0,0000 %) | 0 | 18,46 % |
