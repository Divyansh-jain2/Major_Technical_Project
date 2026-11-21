# Derives accuracy, F1, false-alarm rate, HSS and balanced accuracy for Table "All evaluation measures"
# from the stored precision/recall/CSI/PR-AUC of results.csv (step12/step13 ensembles) and the test counts.
NE,N=37450,1932437; NN=N-NE
rows=[("Persistence (no model)",0.553,0.602,0.405,0.5864,0.1907,0.5651),
("Real data only",0.654,0.495,0.392,0.5986,0.2346,0.4407),
("SMOTE",0.576,0.594,0.413,0.6041,0.2307,0.5071),
("EVT-tail copies of real extremes",0.629,0.542,0.411,0.6132,0.2389,0.5304),
("Diffusion + EVT tail",0.618,0.541,0.405,0.6088,0.2284,0.5792),
("TVAE + EVT tail",0.634,0.511,0.394,0.6102,0.2200,0.6012),
("CTGAN + EVT tail",0.600,0.550,0.403,0.6106,0.2226,0.6014)]
print("never acc",1-NE/N)
for n,P,R,c,a,o,s in rows:
    TP=R*NE; FN=NE-TP; FP=TP*(1/P-1); TN=NN-FP
    acc=(TP+TN)/N; f1=2*P*R/(P+R); far=FP/NN; bal=(R+1-far)/2
    hss=2*(TP*TN-FP*FN)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))
    print(f"{n:34s} acc {acc*100:.2f} P {P:.3f} R {R:.3f} F1 {f1:.3f} CSI {c:.3f} FAR {far*100:.2f} bal {bal:.3f} HSS {hss:.3f} PR {a:.3f} on {o:.3f} sev {s:.3f}")
