// quick_check.C -- sanity numbers for one job (all events_c*.root in a directory)
// usage: root -l -b -q 'quick_check.C("/path/to/jobdir", 661.657, 600)'
#include <TChain.h>
#include <TH1D.h>
#include <iostream>
#include <string>
int quick_check(const char* dir, double eline_keV, double tmeas_s) {
  TChain c("events");
  int nf = c.Add((std::string(dir) + "/events_c*.root").c_str());
  Long64_t n = c.GetEntries();
  TH1D h("h", "", 3000, 0, 3000);
  c.Draw("edep_keV>>h", "", "goff");
  double tmin = c.GetMinimum("time_s"), tmax = c.GetMaximum("time_s");
  int b0 = h.FindBin(eline_keV - 1.0), b1 = h.FindBin(eline_keV + 1.0);
  double pk = h.Integral(b0, b1);
  h.GetXaxis()->SetRangeUser(20, 3000);
  std::cout << "CHECK files=" << nf << " events=" << n << " rate_cps=" << n / tmeas_s
            << " t_range=[" << tmin << "," << tmax << "] s"
            << " FEP(+-1keV@" << eline_keV << ")=" << pk << " FEP_frac=" << (n ? pk / n : 0)
            << " mode_bin_keV(>20)=" << h.GetBinCenter(h.GetMaximumBin()) << std::endl;
  return 0;
}
