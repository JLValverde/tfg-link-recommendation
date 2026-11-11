# Tabla de baselines (T1.11; fusión RRF en T2.1)

Media [IC 95 %] sobre las 1800 consultas de la muestra principal que no son de desarrollo. Generada por `experimentos/tabla_baselines.py` a partir de los `*_resumen.json` y `metricas_*.json`.
Las heurísticas estructurales y la fusión solo se evalúan en caliente; en frío, «no aplica». El modo aleatorio se muestra con la semilla de candidatos 1.

## Modo recuperador (las 50 candidatas del recuperador)

### caliente · tarea completa

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,056 [0,045, 0,066] | 0,065 [0,059, 0,071] | 0,154 [0,142, 0,165] | 0,231 [0,212, 0,249] |
| oráculo (control) | 0,795 [0,776, 0,812] | 0,430 [0,415, 0,444] | 0,795 [0,776, 0,812] | 0,795 [0,776, 0,812] |
| C0 | 0,332 [0,309, 0,354] | 0,207 [0,196, 0,217] | 0,434 [0,413, 0,453] | 0,551 [0,528, 0,574] |
| BM25 | 0,329 [0,307, 0,350] | 0,215 [0,204, 0,225] | 0,443 [0,423, 0,461] | 0,589 [0,566, 0,611] |
| vecinos comunes (B) | 0,425 [0,400, 0,447] | 0,261 [0,249, 0,274] | 0,522 [0,500, 0,542] | 0,637 [0,613, 0,659] |
| Adamic-Adar (B) | 0,452 [0,427, 0,477] | 0,278 [0,266, 0,290] | 0,547 [0,525, 0,567] | 0,664 [0,642, 0,686] |
| vecinos comunes (A) | 0,459 [0,434, 0,483] | 0,265 [0,252, 0,278] | 0,540 [0,518, 0,562] | 0,632 [0,608, 0,655] |
| Adamic-Adar (A) | 0,478 [0,454, 0,503] | 0,280 [0,267, 0,293] | 0,560 [0,538, 0,583] | 0,662 [0,638, 0,683] |
| fusión RRF BM25 + Adamic-Adar (B) | 0,431 [0,407, 0,453] | 0,269 [0,257, 0,280] | 0,536 [0,515, 0,554] | 0,667 [0,643, 0,688] |

### caliente · tarea implícita

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,040 [0,031, 0,049] | 0,053 [0,047, 0,059] | 0,116 [0,106, 0,126] | 0,176 [0,159, 0,194] |
| oráculo (control) | 0,646 [0,624, 0,667] | 0,335 [0,321, 0,350] | 0,646 [0,624, 0,667] | 0,646 [0,624, 0,667] |
| C0 | 0,213 [0,194, 0,231] | 0,149 [0,140, 0,160] | 0,299 [0,281, 0,316] | 0,388 [0,366, 0,409] |
| BM25 | 0,211 [0,191, 0,230] | 0,155 [0,145, 0,165] | 0,304 [0,286, 0,321] | 0,416 [0,392, 0,437] |
| vecinos comunes (B) | 0,287 [0,266, 0,308] | 0,193 [0,181, 0,204] | 0,372 [0,352, 0,391] | 0,473 [0,448, 0,496] |
| Adamic-Adar (B) | 0,311 [0,289, 0,331] | 0,206 [0,194, 0,217] | 0,394 [0,373, 0,414] | 0,496 [0,472, 0,519] |
| vecinos comunes (A) | 0,325 [0,303, 0,346] | 0,196 [0,183, 0,208] | 0,394 [0,374, 0,415] | 0,470 [0,446, 0,493] |
| Adamic-Adar (A) | 0,339 [0,316, 0,361] | 0,206 [0,193, 0,219] | 0,409 [0,388, 0,430] | 0,490 [0,466, 0,513] |
| fusión RRF BM25 + Adamic-Adar (B) | 0,279 [0,260, 0,299] | 0,194 [0,182, 0,205] | 0,372 [0,353, 0,392] | 0,482 [0,459, 0,504] |

### caliente · tarea implícita solo ancla

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,042 [0,033, 0,052] | 0,053 [0,048, 0,060] | 0,119 [0,110, 0,130] | 0,178 [0,161, 0,196] |
| oráculo (control) | 0,654 [0,632, 0,674] | 0,339 [0,324, 0,354] | 0,654 [0,632, 0,674] | 0,654 [0,632, 0,674] |
| C0 | 0,217 [0,199, 0,236] | 0,151 [0,142, 0,162] | 0,304 [0,287, 0,322] | 0,394 [0,373, 0,416] |
| BM25 | 0,214 [0,195, 0,233] | 0,157 [0,147, 0,168] | 0,309 [0,291, 0,325] | 0,423 [0,400, 0,443] |
| vecinos comunes (B) | 0,292 [0,271, 0,313] | 0,196 [0,185, 0,208] | 0,379 [0,359, 0,398] | 0,481 [0,457, 0,504] |
| Adamic-Adar (B) | 0,318 [0,297, 0,338] | 0,209 [0,198, 0,222] | 0,401 [0,382, 0,422] | 0,506 [0,481, 0,528] |
| vecinos comunes (A) | 0,329 [0,308, 0,351] | 0,198 [0,186, 0,211] | 0,401 [0,380, 0,421] | 0,480 [0,456, 0,503] |
| Adamic-Adar (A) | 0,344 [0,321, 0,366] | 0,210 [0,196, 0,223] | 0,416 [0,394, 0,436] | 0,501 [0,476, 0,524] |
| fusión RRF BM25 + Adamic-Adar (B) | 0,288 [0,268, 0,308] | 0,197 [0,185, 0,208] | 0,382 [0,362, 0,400] | 0,493 [0,469, 0,514] |

### frío · tarea completa

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,117 [0,102, 0,133] | 0,108 [0,100, 0,117] | 0,250 [0,235, 0,265] | 0,372 [0,349, 0,393] |
| oráculo (control) | 0,908 [0,894, 0,922] | 0,630 [0,615, 0,645] | 0,908 [0,894, 0,922] | 0,908 [0,894, 0,922] |
| C0 | 0,542 [0,518, 0,566] | 0,339 [0,326, 0,352] | 0,637 [0,616, 0,657] | 0,756 [0,734, 0,777] |
| BM25 | 0,533 [0,509, 0,556] | 0,347 [0,334, 0,360] | 0,642 [0,623, 0,660] | 0,786 [0,767, 0,804] |
| vecinos comunes (B) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |
| vecinos comunes (A) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (A) | no aplica | no aplica | no aplica | no aplica |
| fusión RRF BM25 + Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |

### frío · tarea implícita

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,081 [0,069, 0,093] | 0,078 [0,071, 0,085] | 0,183 [0,170, 0,195] | 0,271 [0,251, 0,289] |
| oráculo (control) | 0,774 [0,755, 0,791] | 0,445 [0,428, 0,461] | 0,774 [0,755, 0,791] | 0,774 [0,755, 0,791] |
| C0 | 0,339 [0,317, 0,363] | 0,215 [0,204, 0,227] | 0,435 [0,415, 0,455] | 0,543 [0,520, 0,565] |
| BM25 | 0,339 [0,317, 0,361] | 0,219 [0,208, 0,230] | 0,441 [0,422, 0,460] | 0,568 [0,546, 0,589] |
| vecinos comunes (B) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |
| vecinos comunes (A) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (A) | no aplica | no aplica | no aplica | no aplica |
| fusión RRF BM25 + Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |

### frío · tarea implícita solo ancla

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,083 [0,070, 0,096] | 0,080 [0,072, 0,087] | 0,186 [0,173, 0,199] | 0,276 [0,256, 0,296] |
| oráculo (control) | 0,782 [0,763, 0,799] | 0,451 [0,435, 0,466] | 0,782 [0,763, 0,799] | 0,782 [0,763, 0,799] |
| C0 | 0,347 [0,324, 0,371] | 0,219 [0,208, 0,231] | 0,443 [0,422, 0,464] | 0,551 [0,527, 0,573] |
| BM25 | 0,343 [0,321, 0,365] | 0,224 [0,212, 0,235] | 0,447 [0,428, 0,466] | 0,578 [0,556, 0,600] |
| vecinos comunes (B) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |
| vecinos comunes (A) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (A) | no aplica | no aplica | no aplica | no aplica |
| fusión RRF BM25 + Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |

## Modo aleatorio (semilla 1)

### caliente · tarea completa

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,122 [0,106, 0,137] | 0,132 [0,124, 0,139] | 0,284 [0,271, 0,298] | 0,449 [0,427, 0,473] |
| oráculo (control) | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| C0 | 0,848 [0,831, 0,864] | 0,692 [0,679, 0,705] | 0,900 [0,888, 0,911] | 0,961 [0,952, 0,969] |
| BM25 | 0,864 [0,847, 0,879] | 0,702 [0,691, 0,714] | 0,914 [0,903, 0,924] | 0,978 [0,971, 0,984] |
| vecinos comunes (B) | 0,919 [0,906, 0,931] | 0,801 [0,790, 0,813] | 0,948 [0,940, 0,957] | 0,984 [0,978, 0,989] |
| Adamic-Adar (B) | 0,931 [0,919, 0,943] | 0,822 [0,811, 0,832] | 0,956 [0,948, 0,964] | 0,987 [0,982, 0,992] |
| vecinos comunes (A) | 0,922 [0,911, 0,935] | 0,827 [0,817, 0,837] | 0,952 [0,945, 0,960] | 0,987 [0,982, 0,992] |
| Adamic-Adar (A) | 0,932 [0,919, 0,944] | 0,849 [0,839, 0,859] | 0,958 [0,950, 0,965] | 0,989 [0,984, 0,993] |
| fusión RRF BM25 + Adamic-Adar (B) | 0,932 [0,920, 0,944] | 0,808 [0,797, 0,818] | 0,956 [0,948, 0,964] | 0,987 [0,982, 0,992] |

### caliente · tarea implícita

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,099 [0,086, 0,113] | 0,120 [0,112, 0,128] | 0,248 [0,235, 0,261] | 0,380 [0,358, 0,402] |
| oráculo (control) | 1,000 [1,000, 1,000] | 0,999 [0,999, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| C0 | 0,747 [0,728, 0,767] | 0,636 [0,621, 0,649] | 0,827 [0,813, 0,841] | 0,927 [0,914, 0,938] |
| BM25 | 0,740 [0,720, 0,759] | 0,632 [0,618, 0,645] | 0,829 [0,815, 0,842] | 0,941 [0,930, 0,951] |
| vecinos comunes (B) | 0,869 [0,853, 0,885] | 0,768 [0,756, 0,780] | 0,914 [0,903, 0,924] | 0,969 [0,962, 0,977] |
| Adamic-Adar (B) | 0,889 [0,874, 0,904] | 0,790 [0,778, 0,801] | 0,927 [0,917, 0,937] | 0,973 [0,966, 0,981] |
| vecinos comunes (A) | 0,873 [0,858, 0,889] | 0,797 [0,785, 0,808] | 0,921 [0,911, 0,931] | 0,977 [0,970, 0,984] |
| Adamic-Adar (A) | 0,889 [0,874, 0,904] | 0,819 [0,808, 0,830] | 0,932 [0,922, 0,941] | 0,982 [0,976, 0,988] |
| fusión RRF BM25 + Adamic-Adar (B) | 0,864 [0,848, 0,880] | 0,758 [0,746, 0,770] | 0,912 [0,901, 0,922] | 0,971 [0,963, 0,979] |

### caliente · tarea implícita solo ancla

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,099 [0,086, 0,113] | 0,120 [0,112, 0,128] | 0,248 [0,236, 0,262] | 0,382 [0,361, 0,404] |
| oráculo (control) | 1,000 [1,000, 1,000] | 0,999 [0,999, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| C0 | 0,755 [0,735, 0,774] | 0,638 [0,624, 0,652] | 0,832 [0,818, 0,847] | 0,929 [0,917, 0,941] |
| BM25 | 0,747 [0,727, 0,767] | 0,635 [0,621, 0,649] | 0,833 [0,820, 0,847] | 0,941 [0,930, 0,951] |
| vecinos comunes (B) | 0,874 [0,858, 0,889] | 0,770 [0,758, 0,782] | 0,917 [0,906, 0,927] | 0,969 [0,962, 0,977] |
| Adamic-Adar (B) | 0,892 [0,877, 0,907] | 0,792 [0,780, 0,804] | 0,929 [0,919, 0,939] | 0,974 [0,967, 0,981] |
| vecinos comunes (A) | 0,877 [0,862, 0,893] | 0,798 [0,787, 0,810] | 0,924 [0,914, 0,934] | 0,977 [0,970, 0,984] |
| Adamic-Adar (A) | 0,893 [0,878, 0,908] | 0,820 [0,809, 0,831] | 0,933 [0,924, 0,943] | 0,982 [0,976, 0,988] |
| fusión RRF BM25 + Adamic-Adar (B) | 0,868 [0,852, 0,884] | 0,760 [0,748, 0,772] | 0,914 [0,903, 0,924] | 0,971 [0,962, 0,978] |

### frío · tarea completa

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,175 [0,158, 0,192] | 0,172 [0,164, 0,179] | 0,370 [0,354, 0,384] | 0,609 [0,587, 0,631] |
| oráculo (control) | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| C0 | 0,933 [0,921, 0,944] | 0,783 [0,773, 0,793] | 0,958 [0,951, 0,966] | 0,991 [0,987, 0,995] |
| BM25 | 0,942 [0,931, 0,952] | 0,796 [0,787, 0,805] | 0,966 [0,960, 0,972] | 0,996 [0,993, 0,999] |
| vecinos comunes (B) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |
| vecinos comunes (A) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (A) | no aplica | no aplica | no aplica | no aplica |
| fusión RRF BM25 + Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |

### frío · tarea implícita

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,130 [0,114, 0,145] | 0,137 [0,130, 0,145] | 0,304 [0,289, 0,317] | 0,494 [0,472, 0,518] |
| oráculo (control) | 1,000 [1,000, 1,000] | 0,992 [0,990, 0,994] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| C0 | 0,832 [0,814, 0,849] | 0,657 [0,645, 0,669] | 0,892 [0,880, 0,903] | 0,969 [0,961, 0,977] |
| BM25 | 0,826 [0,807, 0,842] | 0,654 [0,643, 0,666] | 0,892 [0,881, 0,902] | 0,976 [0,969, 0,982] |
| vecinos comunes (B) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |
| vecinos comunes (A) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (A) | no aplica | no aplica | no aplica | no aplica |
| fusión RRF BM25 + Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |

### frío · tarea implícita solo ancla

| Sistema | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|
| aleatorio (control) | 0,131 [0,115, 0,146] | 0,139 [0,131, 0,146] | 0,306 [0,291, 0,319] | 0,501 [0,479, 0,524] |
| oráculo (control) | 1,000 [1,000, 1,000] | 0,992 [0,990, 0,995] | 1,000 [1,000, 1,000] | 1,000 [1,000, 1,000] |
| C0 | 0,840 [0,823, 0,857] | 0,662 [0,650, 0,674] | 0,897 [0,886, 0,908] | 0,972 [0,964, 0,979] |
| BM25 | 0,833 [0,816, 0,849] | 0,660 [0,648, 0,671] | 0,896 [0,886, 0,907] | 0,977 [0,971, 0,983] |
| vecinos comunes (B) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |
| vecinos comunes (A) | no aplica | no aplica | no aplica | no aplica |
| Adamic-Adar (A) | no aplica | no aplica | no aplica | no aplica |
| fusión RRF BM25 + Adamic-Adar (B) | no aplica | no aplica | no aplica | no aplica |

## Diferencias pareadas, modo recuperador

Diferencia media A − B por consulta [IC 95 % por bootstrap pareado, 1000 remuestreos], sobre las 1800 consultas. Con * las que no incluyen el 0.

### Tarea completa

| Comparación | Escenario | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|---|
| BM25 − C0 | caliente | −0,002 [−0,023, +0,019] | +0,008 [+0,001, +0,016] * | +0,009 [−0,006, +0,024] | +0,038 [+0,019, +0,058] * |
| BM25 − C0 | frío | −0,009 [−0,031, +0,013] | +0,008 [−0,001, +0,016] | +0,005 [−0,011, +0,021] | +0,030 [+0,013, +0,046] * |
| vecinos comunes (B) − C0 | caliente | +0,093 [+0,068, +0,119] * | +0,055 [+0,045, +0,065] * | +0,087 [+0,068, +0,109] * | +0,086 [+0,061, +0,111] * |
| Adamic-Adar (B) − C0 | caliente | +0,120 [+0,094, +0,147] * | +0,071 [+0,062, +0,081] * | +0,113 [+0,093, +0,134] * | +0,113 [+0,091, +0,137] * |
| vecinos comunes (A) − vecinos comunes (B) | caliente | +0,034 [+0,017, +0,050] * | +0,004 [−0,002, +0,011] | +0,019 [+0,007, +0,030] * | −0,005 [−0,018, +0,008] |
| Adamic-Adar (A) − Adamic-Adar (B) | caliente | +0,027 [+0,009, +0,044] * | +0,003 [−0,003, +0,009] | +0,013 [+0,001, +0,024] * | −0,003 [−0,015, +0,010] |
| fusión RRF BM25 + Adamic-Adar (B) − Adamic-Adar (B) | caliente | −0,021 [−0,044, +0,001] | −0,009 [−0,016, −0,002] * | −0,011 [−0,027, +0,004] | +0,003 [−0,013, +0,018] |
| fusión RRF BM25 + Adamic-Adar (B) − BM25 | caliente | +0,101 [+0,079, +0,124] * | +0,054 [+0,047, +0,061] * | +0,093 [+0,078, +0,108] * | +0,078 [+0,060, +0,094] * |

### Tarea implícita

| Comparación | Escenario | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|---|
| BM25 − C0 | caliente | −0,002 [−0,019, +0,016] | +0,006 [−0,002, +0,013] | +0,005 [−0,008, +0,019] | +0,028 [+0,008, +0,047] * |
| BM25 − C0 | frío | −0,001 [−0,021, +0,021] | +0,004 [−0,004, +0,011] | +0,006 [−0,009, +0,023] | +0,024 [+0,007, +0,043] * |
| vecinos comunes (B) − C0 | caliente | +0,074 [+0,052, +0,096] * | +0,043 [+0,033, +0,053] * | +0,073 [+0,055, +0,092] * | +0,085 [+0,062, +0,108] * |
| Adamic-Adar (B) − C0 | caliente | +0,098 [+0,077, +0,119] * | +0,056 [+0,046, +0,066] * | +0,095 [+0,077, +0,113] * | +0,108 [+0,086, +0,132] * |
| vecinos comunes (A) − vecinos comunes (B) | caliente | +0,038 [+0,021, +0,054] * | +0,003 [−0,003, +0,009] | +0,022 [+0,010, +0,034] * | −0,003 [−0,016, +0,012] |
| Adamic-Adar (A) − Adamic-Adar (B) | caliente | +0,028 [+0,013, +0,044] * | +0,000 [−0,006, +0,007] | +0,016 [+0,004, +0,027] * | −0,006 [−0,020, +0,007] |
| fusión RRF BM25 + Adamic-Adar (B) − Adamic-Adar (B) | caliente | −0,031 [−0,052, −0,012] * | −0,012 [−0,019, −0,005] * | −0,021 [−0,036, −0,007] * | −0,014 [−0,031, +0,002] |
| fusión RRF BM25 + Adamic-Adar (B) − BM25 | caliente | +0,068 [+0,051, +0,088] * | +0,039 [+0,031, +0,046] * | +0,068 [+0,056, +0,082] * | +0,067 [+0,050, +0,084] * |

### Tarea implícita solo ancla

| Comparación | Escenario | precision@1 | precision_norm@5 | MRR | hits@5 |
|---|---|---|---|---|---|
| BM25 − C0 | caliente | −0,003 [−0,020, +0,015] | +0,006 [−0,002, +0,013] | +0,004 [−0,009, +0,019] | +0,029 [+0,009, +0,048] * |
| BM25 − C0 | frío | −0,004 [−0,025, +0,017] | +0,004 [−0,003, +0,012] | +0,004 [−0,011, +0,021] | +0,028 [+0,009, +0,047] * |
| vecinos comunes (B) − C0 | caliente | +0,075 [+0,052, +0,097] * | +0,045 [+0,035, +0,055] * | +0,074 [+0,057, +0,093] * | +0,087 [+0,064, +0,111] * |
| Adamic-Adar (B) − C0 | caliente | +0,101 [+0,079, +0,124] * | +0,058 [+0,048, +0,067] * | +0,097 [+0,080, +0,116] * | +0,111 [+0,089, +0,135] * |
| vecinos comunes (A) − vecinos comunes (B) | caliente | +0,037 [+0,021, +0,053] * | +0,002 [−0,004, +0,008] | +0,022 [+0,010, +0,034] * | −0,001 [−0,015, +0,013] |
| Adamic-Adar (A) − Adamic-Adar (B) | caliente | +0,026 [+0,010, +0,042] * | +0,000 [−0,006, +0,007] | +0,015 [+0,003, +0,026] * | −0,004 [−0,019, +0,008] |
| fusión RRF BM25 + Adamic-Adar (B) − Adamic-Adar (B) | caliente | −0,029 [−0,049, −0,010] * | −0,013 [−0,020, −0,006] * | −0,020 [−0,034, −0,006] * | −0,012 [−0,029, +0,004] |
| fusión RRF BM25 + Adamic-Adar (B) − BM25 | caliente | +0,074 [+0,056, +0,093] * | +0,040 [+0,032, +0,047] * | +0,073 [+0,060, +0,086] * | +0,070 [+0,053, +0,087] * |

