# Baselines estructurales sobre A y sobre B (caliente)

Media [IC 95 %] sobre las 1800 consultas. B son los enlaces de WikiCS escritos en el artículo de origen (de donde salen los enlaces de referencia); A, todos los enlaces de WikiCS, incluidos los de plantillas de navegación. Solo en caliente: en frío no aplican. Las diferencias A − B son pareadas, en modo recuperador. Generada por `experimentos/tablas_finales.py`.

## Modo recuperador

### Tarea completa

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| vecinos comunes (B) | 0,425 [0,400, 0,447] | 0,261 [0,249, 0,274] | 0,522 [0,500, 0,542] | 0,637 [0,613, 0,659] |
| vecinos comunes (A) | 0,459 [0,434, 0,483] | 0,265 [0,252, 0,278] | 0,540 [0,518, 0,562] | 0,632 [0,608, 0,655] |
| Adamic-Adar (B) | 0,452 [0,427, 0,477] | 0,278 [0,266, 0,290] | 0,547 [0,525, 0,567] | 0,664 [0,642, 0,686] |
| Adamic-Adar (A) | 0,478 [0,454, 0,503] | 0,280 [0,267, 0,293] | 0,560 [0,538, 0,583] | 0,662 [0,638, 0,683] |

### Tarea implícita

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| vecinos comunes (B) | 0,287 [0,266, 0,308] | 0,193 [0,181, 0,204] | 0,372 [0,352, 0,391] | 0,473 [0,448, 0,496] |
| vecinos comunes (A) | 0,325 [0,303, 0,346] | 0,196 [0,183, 0,208] | 0,394 [0,374, 0,415] | 0,470 [0,446, 0,493] |
| Adamic-Adar (B) | 0,311 [0,289, 0,331] | 0,206 [0,194, 0,217] | 0,394 [0,373, 0,414] | 0,496 [0,472, 0,519] |
| Adamic-Adar (A) | 0,339 [0,316, 0,361] | 0,206 [0,193, 0,219] | 0,409 [0,388, 0,430] | 0,490 [0,466, 0,513] |

### Tarea implícita solo ancla

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| vecinos comunes (B) | 0,292 [0,271, 0,313] | 0,196 [0,185, 0,208] | 0,379 [0,359, 0,398] | 0,481 [0,457, 0,504] |
| vecinos comunes (A) | 0,329 [0,308, 0,351] | 0,198 [0,186, 0,211] | 0,401 [0,380, 0,421] | 0,480 [0,456, 0,503] |
| Adamic-Adar (B) | 0,318 [0,297, 0,338] | 0,209 [0,198, 0,222] | 0,401 [0,382, 0,422] | 0,506 [0,481, 0,528] |
| Adamic-Adar (A) | 0,344 [0,321, 0,366] | 0,210 [0,196, 0,223] | 0,416 [0,394, 0,436] | 0,501 [0,476, 0,524] |

## Modo aleatorio (semilla 1)

### Tarea completa

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| vecinos comunes (B) | 0,919 [0,906, 0,931] | 0,801 [0,790, 0,813] | 0,948 [0,940, 0,957] | 0,984 [0,978, 0,989] |
| vecinos comunes (A) | 0,922 [0,911, 0,935] | 0,827 [0,817, 0,837] | 0,952 [0,945, 0,960] | 0,987 [0,982, 0,992] |
| Adamic-Adar (B) | 0,931 [0,919, 0,943] | 0,822 [0,811, 0,832] | 0,956 [0,948, 0,964] | 0,987 [0,982, 0,992] |
| Adamic-Adar (A) | 0,932 [0,919, 0,944] | 0,849 [0,839, 0,859] | 0,958 [0,950, 0,965] | 0,989 [0,984, 0,993] |

### Tarea implícita

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| vecinos comunes (B) | 0,869 [0,853, 0,885] | 0,768 [0,756, 0,780] | 0,914 [0,903, 0,924] | 0,969 [0,962, 0,977] |
| vecinos comunes (A) | 0,873 [0,858, 0,889] | 0,797 [0,785, 0,808] | 0,921 [0,911, 0,931] | 0,977 [0,970, 0,984] |
| Adamic-Adar (B) | 0,889 [0,874, 0,904] | 0,790 [0,778, 0,801] | 0,927 [0,917, 0,937] | 0,973 [0,966, 0,981] |
| Adamic-Adar (A) | 0,889 [0,874, 0,904] | 0,819 [0,808, 0,830] | 0,932 [0,922, 0,941] | 0,982 [0,976, 0,988] |

### Tarea implícita solo ancla

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| vecinos comunes (B) | 0,874 [0,858, 0,889] | 0,770 [0,758, 0,782] | 0,917 [0,906, 0,927] | 0,969 [0,962, 0,977] |
| vecinos comunes (A) | 0,877 [0,862, 0,893] | 0,798 [0,787, 0,810] | 0,924 [0,914, 0,934] | 0,977 [0,970, 0,984] |
| Adamic-Adar (B) | 0,892 [0,877, 0,907] | 0,792 [0,780, 0,804] | 0,929 [0,919, 0,939] | 0,974 [0,967, 0,981] |
| Adamic-Adar (A) | 0,893 [0,878, 0,908] | 0,820 [0,809, 0,831] | 0,933 [0,924, 0,943] | 0,982 [0,976, 0,988] |

## Diferencias A − B, modo recuperador

Con * las que no incluyen el 0.

| Tarea | Comparación | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|---|
| completa | vecinos comunes (A) − vecinos comunes (B) | +0,034 [+0,017, +0,050] * | +0,004 [−0,002, +0,011] | +0,019 [+0,007, +0,030] * | −0,005 [−0,018, +0,008] |
| completa | Adamic-Adar (A) − Adamic-Adar (B) | +0,027 [+0,009, +0,044] * | +0,003 [−0,003, +0,009] | +0,013 [+0,001, +0,024] * | −0,003 [−0,015, +0,010] |
| implícita | vecinos comunes (A) − vecinos comunes (B) | +0,038 [+0,021, +0,054] * | +0,003 [−0,003, +0,009] | +0,022 [+0,010, +0,034] * | −0,003 [−0,016, +0,012] |
| implícita | Adamic-Adar (A) − Adamic-Adar (B) | +0,028 [+0,013, +0,044] * | +0,000 [−0,006, +0,007] | +0,016 [+0,004, +0,027] * | −0,006 [−0,020, +0,007] |
| implícita solo ancla | vecinos comunes (A) − vecinos comunes (B) | +0,037 [+0,021, +0,053] * | +0,002 [−0,004, +0,008] | +0,022 [+0,010, +0,034] * | −0,001 [−0,015, +0,013] |
| implícita solo ancla | Adamic-Adar (A) − Adamic-Adar (B) | +0,026 [+0,010, +0,042] * | +0,000 [−0,006, +0,007] | +0,015 [+0,003, +0,026] * | −0,004 [−0,019, +0,008] |
