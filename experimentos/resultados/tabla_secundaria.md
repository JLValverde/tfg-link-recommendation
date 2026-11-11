# Tabla secundaria: modo aleatorio (semilla 1, tarea implícita)

Media [IC 95 %] sobre las 1800 consultas, con las 50 candidatas del modo aleatorio (enlaces de referencia, como mucho G = 10, más negativos al azar): mide la capacidad de ordenar, aislada, y no se mezcla con el modo recuperador. Las semillas 2 y 3 están en los `*_resumen.json`; las tres tareas, en `tabla_principal.md`. C1, C2 y las réplicas no se han ejecutado en este modo (recortado). Generada por `experimentos/tablas_finales.py`.

| Sistema | frío: precision@1 | frío: precision_norm@5 | frío: MRR | frío: hits@5 | caliente: precision@1 | caliente: precision_norm@5 | caliente: MRR | caliente: hits@5 |
|---|---|---|---|---|---|---|---|---|
| aleatorio (control) | 0,130 [0,114, 0,145] | 0,137 [0,130, 0,145] | 0,304 [0,289, 0,317] | 0,494 [0,472, 0,518] | 0,099 [0,086, 0,113] | 0,120 [0,112, 0,128] | 0,248 [0,235, 0,261] | 0,380 [0,358, 0,402] |
| oráculo (control) | 1,000 [1,000, 1,000] | 0,992 [0,990, 0,994] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] | 0,999 [0,999, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| C0 | 0,832 [0,814, 0,849] | 0,657 [0,645, 0,669] | 0,892 [0,880, 0,903] | 0,969 [0,961, 0,977] | 0,747 [0,728, 0,767] | 0,636 [0,621, 0,649] | 0,827 [0,813, 0,841] | 0,927 [0,914, 0,938] |
| BM25 | 0,826 [0,807, 0,842] | 0,654 [0,643, 0,666] | 0,892 [0,881, 0,902] | 0,976 [0,969, 0,982] | 0,740 [0,720, 0,759] | 0,632 [0,618, 0,645] | 0,829 [0,815, 0,842] | 0,941 [0,930, 0,951] |
| vecinos comunes (B) | no aplica | no aplica | no aplica | no aplica | 0,869 [0,853, 0,885] | 0,768 [0,756, 0,780] | 0,914 [0,903, 0,924] | 0,969 [0,962, 0,977] |
| Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica | 0,889 [0,874, 0,904] | 0,790 [0,778, 0,801] | 0,927 [0,917, 0,937] | 0,973 [0,966, 0,981] |
| vecinos comunes (A) | no aplica | no aplica | no aplica | no aplica | 0,873 [0,858, 0,889] | 0,797 [0,785, 0,808] | 0,921 [0,911, 0,931] | 0,977 [0,970, 0,984] |
| Adamic-Adar (A) | no aplica | no aplica | no aplica | no aplica | 0,889 [0,874, 0,904] | 0,819 [0,808, 0,830] | 0,932 [0,922, 0,941] | 0,982 [0,976, 0,988] |
| fusión RRF BM25 + Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica | 0,864 [0,848, 0,880] | 0,758 [0,746, 0,770] | 0,912 [0,901, 0,922] | 0,971 [0,963, 0,979] |
| C1 (qwen2.5:3b) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) |
| C1 (qwen2.5:7b) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) |
| C1 (qwen2.5:14b) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) |
| C2 (qwen2.5:3b) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) |
| C2 (qwen2.5:7b) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) |
| C2 (qwen2.5:14b) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) | no ejecutado (recortado) |
