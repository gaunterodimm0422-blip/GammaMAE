// compress_hits.C -- GATE Hits (one row per step) -> one row per event
// usage: root -l -b -q 'compress_hits.C("hits_c0.root","events_c0.root")'
// output tree "events": edep_keV (F), time_s (D, first hit), sourceID (S), nhits (I)
#include <TFile.h>
#include <TTree.h>
#include <TLeaf.h>
#include <TKey.h>
#include <TList.h>
#include <TParameter.h>
#include <TSystem.h>
#include <iostream>
#include <vector>
#include <string>

int compress_hits(const char* fin, const char* fout) {
  TFile f(fin, "READ");
  if (f.IsZombie()) { std::cerr << "COMPRESS_ERR cannot open " << fin << std::endl; return 2; }
  std::vector<std::string> trees;
  TIter next(f.GetListOfKeys());
  while (TKey* k = (TKey*)next()) {
    std::string nm = k->GetName(), cl = k->GetClassName();
    if (cl == "TTree" && nm.rfind("Hits", 0) == 0) {
      bool dup = false; for (auto& s : trees) if (s == nm) dup = true;
      if (!dup) trees.push_back(nm);
    }
  }
  if (trees.empty()) { std::cerr << "COMPRESS_ERR no Hits tree in " << fin << std::endl; return 3; }

  TFile g(fout, "RECREATE", "", 505);
  TTree ev("events", "GATE events: summed edep per (runID,eventID)");
  Float_t edep_keV; Double_t time_s; Short_t src; Int_t nh;
  ev.Branch("edep_keV", &edep_keV, "edep_keV/F");
  ev.Branch("time_s", &time_s, "time_s/D");
  ev.Branch("sourceID", &src, "sourceID/S");
  ev.Branch("nhits", &nh, "nhits/I");

  Long64_t nhits_tot = 0, nev = 0, nev_zero = 0;
  for (auto& tn : trees) {
    TTree* t = (TTree*)f.Get(tn.c_str());
    const char* need[5] = {"eventID", "runID", "edep", "time", "sourceID"};
    t->SetBranchStatus("*", 0);
    TLeaf* L[5];
    for (int i = 0; i < 5; i++) {
      t->SetBranchStatus(need[i], 1);
      L[i] = t->GetLeaf(need[i]);
      if (!L[i]) { std::cerr << "COMPRESS_ERR missing branch " << need[i] << " in " << tn << std::endl; return 4; }
    }
    Long64_t n = t->GetEntries();
    nhits_tot += n;
    Long64_t ce = -1, cr = -1; double sum = 0, t0 = 0; int s = -1, cnt = 0; bool open = false;
    auto flush = [&]() {
      if (!open) return;
      if (sum > 0) { edep_keV = (Float_t)(sum * 1000.0); time_s = t0; src = (Short_t)s; nh = cnt; ev.Fill(); nev++; }
      else nev_zero++;
      open = false;
    };
    for (Long64_t i = 0; i < n; i++) {
      t->GetEntry(i);
      Long64_t e = (Long64_t)L[0]->GetValue(), r = (Long64_t)L[1]->GetValue();
      double ed = L[2]->GetValue(), tm = L[3]->GetValue();
      int sid = (int)L[4]->GetValue();
      if (open && (e != ce || r != cr)) flush();
      if (!open) { ce = e; cr = r; sum = 0; t0 = tm; s = sid; cnt = 0; open = true; }
      sum += ed; cnt++;
      if (tm < t0) t0 = tm;
    }
    flush();
  }
  g.cd();
  ev.Write();
  TParameter<Long64_t>("n_hits", nhits_tot).Write();
  TParameter<Long64_t>("n_events", nev).Write();
  g.Close();
  std::cout << "COMPRESS_OK hits=" << nhits_tot << " events=" << nev << " zero_edep_events=" << nev_zero << std::endl;
  return 0;
}
