// higgs4mu_select.cpp (ISO forzado + config flexible + diagnósticos) [CORREGIDO]
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <cstdio>
#include <tuple>
#include <array>
#include <iostream>
#include <map>
#include <sstream>
#include <cstdlib>

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TBranch.h"
#include "TH1.h"
#include "TLorentzVector.h"

// ===================== Física base =====================
static constexpr double MZ  = 91.1876;        // GeV
static constexpr double Mmu = 0.1056583745;   // GeV

// ===================== Config en runtime =====================
struct Config {
  // Cortes básicos por muón
  double ETA_MAX        = 2.4;    // |eta| < ETA_MAX
  double PT_MIN_BASE    = 5.0;    // pT > PT_MIN_BASE
  double DXY_MAX        = 0.5;    // |dxy| < DXY_MAX (cm)
  double DZ_MAX         = 1.0;    // |dz|  < DZ_MAX  (cm)
  double SIP3D_MAX      = 4.0;    // |SIP3D| < SIP3D_MAX

  // ISO
  bool   USE_ISO        = false;  // aplicar ISO si existe la rama
  double ISO_MAX        = 0.40;   // relISO (DB-corr)
  std::string ISO_BRANCH = "muon_pfreliso04DBCorr"; // nombre exacto

  // Requisitos pT en pares / 4µ
  bool   ENFORCE_4MU_PT = false;  // 20/10/7/5 en los 4 muones finales
  double PT_LEAD        = 20.0;
  double PT_SUB         = 10.0;
  double PT_THIRD       = 7.0;
  double PT_FOURTH      = 5.0;

  // Ventanas de masa
  double MZA_MIN = 40.0, MZA_MAX = 120.0;
  double MZB_MIN = 12.0, MZB_MAX = 120.0;

  // Lógica de combinaciones (si quedan >4 muones tras preselect)
  bool   USE_COMB_SEARCH = false;

  // Histogramas de pares
  bool   FILL_PAIR_HIST_BEST_ONLY = true;  // solo las dos parejas elegidas
  bool   FILL_PAIR_HIST_PRE       = false; // diagnóstico: pares OS antes de elegir

  // Posibilidad de desactivar etapas de preselección (diagnóstico)
  bool   APPLY_ETA = true;
  bool   APPLY_ID  = true;  // ≥ Loose
  bool   APPLY_IP  = true;  // |dxy|,|dz|
  bool   APPLY_SIP = true;  // |SIP3D|
};

// Nivel por muón: 3 Tight, 2 Medium, 1 Loose, 0 no clasificado
inline int muLevel(int isLoose, int isMedium, int isTight) {
  if (isTight)  return 3;
  if (isMedium) return 2;
  if (isLoose)  return 1;
  return 0;
}

struct Mu {
  double pt, eta, phi; int q;
  int level; // 3/2/1/0
  int idx;   // índice en el ntuple
};

struct Pairing {
  TLorentzVector Za, Zb;   // Za: par más cercano a MZ; Zb: off-shell
  double mZa, mZb;         // masas ya ordenadas
  int idxZa1, idxZa2;      // índices (Za)
  int idxZb1, idxZb2;      // índices (Zb)
  int pairing_code;        // 0=12+34, 1=13+24, 2=14+23
  // para cortes de pT en Za
  double ptZa1, ptZa2;
};

// Normaliza (A,B) para que Za sea el más cercano a MZ
static Pairing make_cand_normalized(const TLorentzVector& A1, const TLorentzVector& A2,
                                    const TLorentzVector& B1, const TLorentzVector& B2,
                                    int idxA1, int idxA2, int idxB1, int idxB2,
                                    int pairing_code_base) {
  const double mA = (A1+A2).M();
  const double mB = (B1+B2).M();
  if (std::abs(mB - MZ) < std::abs(mA - MZ)) {
    Pairing P;
    P.Za   = B1 + B2;  P.Zb   = A1 + A2;
    P.mZa  = mB;       P.mZb  = mA;
    P.idxZa1 = idxB1;  P.idxZa2 = idxB2;
    P.idxZb1 = idxA1;  P.idxZb2 = idxA2;
    P.pairing_code = pairing_code_base;
    P.ptZa1 = B1.Pt(); P.ptZa2 = B2.Pt();
    return P;
  } else {
    Pairing P;
    P.Za   = A1 + A2;  P.Zb   = B1 + B2;
    P.mZa  = mA;       P.mZb  = mB;
    P.idxZa1 = idxA1;  P.idxZa2 = idxA2;
    P.idxZb1 = idxB1;  P.idxZb2 = idxB2;
    P.pairing_code = pairing_code_base;
    P.ptZa1 = A1.Pt(); P.ptZa2 = A2.Pt();
    return P;
  }
}

static bool pass_4mu_pt(const std::array<double,4>& pts, const Config& cfg) {
  return (pts[0] > cfg.PT_LEAD &&
          pts[1] > cfg.PT_SUB  &&
          pts[2] > cfg.PT_THIRD &&
          pts[3] > cfg.PT_FOURTH);
}

// -------- Histos globales --------
static TH1I* g_cutflow = nullptr;
// Pares (solo del pareo ganador, por defecto)
static TH1F *g_h_mZ12_best=nullptr, *g_h_mZ34_best=nullptr,
            *g_h_mZ13_best=nullptr, *g_h_mZ24_best=nullptr,
            *g_h_mZ14_best=nullptr, *g_h_mZ23_best=nullptr;
// (Opcional) Pares previos a elección (diagnóstico)
static TH1F *g_h_mZ12_pre=nullptr, *g_h_mZ34_pre=nullptr,
            *g_h_mZ13_pre=nullptr, *g_h_mZ24_pre=nullptr,
            *g_h_mZ14_pre=nullptr, *g_h_mZ23_pre=nullptr;

// Finales globales
static TH1F *g_h_mZa=nullptr, *g_h_mZb=nullptr, *g_h_m4mu=nullptr;
// Por categorías tradicionales
static TH1F *g_h_m4mu_AllTight=nullptr, *g_h_m4mu_AllMedium=nullptr, *g_h_m4mu_AllLoose=nullptr;

// Por combinaciones T{nT}M{nM}L{nL}
static std::map<std::string, TH1F*> g_h_m4mu_by_comp;
static std::map<std::string, TH1F*> g_h_mZa_by_comp;
static std::map<std::string, TH1F*> g_h_mZb_by_comp;

static std::string comp_key(int nT, int nM, int nL) {
  std::ostringstream os;
  os << "T" << nT << "M" << nM << "L" << nL;
  return os.str();
}

static TH1F* ensure_hist(std::map<std::string, TH1F*>& dict, const std::string& key,
                         const char* base_title, int nb, double lo, double hi) {
  auto it = dict.find(key);
  if (it != dict.end()) return it->second;
  std::string name = std::string("h_") + base_title + "_" + key;
  std::string title= std::string(base_title) + " (" + key + ");m [GeV];Events";
  TH1F* h = new TH1F(name.c_str(), title.c_str(), nb, lo, hi);
  h->Sumw2();
  dict[key] = h;
  return h;
}

// ---- helpers de 4-vectores y clasificación ----
static inline TLorentzVector p4_of(const Mu& m) {
  TLorentzVector v; v.SetPtEtaPhiM(m.pt, m.eta, m.phi, Mmu); return v;
}

// Categorías tradicionales: 3=AllTight, 2=AllMedium, 1=AllLoose, 0=Unclassified
static int classify_event(const std::vector<Mu>& four, int& nT, int& nM, int& nL) {
  nT=nM=nL=0;
  int mn = 10;
  for (auto& m : four) {
    if (m.level==3) ++nT; else if (m.level==2) ++nM; else if (m.level==1) ++nL;
    mn = std::min(mn, m.level);
  }
  if (mn==3) return 3;                       // 4 Tight
  if (mn==2 && nT<4) return 2;               // todos >= Medium, pero no 4 Tight
  if (mn==1 && (nM+nT)<4) return 1;          // todos >= Loose, pero no todos >= Medium
  return 0;
}

// Mejor pareo OS-OS entre 4 muones dados (ordenados por pT desc localmente)
static bool best_pairing_for_four(const std::array<Mu,4>& mu4,
                                  Pairing& bestP,
                                  std::vector<Mu>& sel4) {
  std::vector<Mu> local = {mu4[0], mu4[1], mu4[2], mu4[3]};
  std::sort(local.begin(), local.end(), [](const Mu& a, const Mu& b){return a.pt>b.pt;});
  const Mu &m1=local[0], &m2=local[1], &m3=local[2], &m4=local[3];
  TLorentzVector v1=p4_of(m1), v2=p4_of(m2), v3=p4_of(m3), v4=p4_of(m4);

  bool os12 = (m1.q + m2.q == 0);
  bool os34 = (m3.q + m4.q == 0);
  bool os13 = (m1.q + m3.q == 0);
  bool os24 = (m2.q + m4.q == 0);
  bool os14 = (m1.q + m4.q == 0);
  bool os23 = (m2.q + m3.q == 0);

  std::vector<Pairing> cands; cands.reserve(3);
  auto add_cand = [&](const TLorentzVector& A1, const TLorentzVector& A2,
                      const TLorentzVector& B1, const TLorentzVector& B2,
                      int idxA1, int idxA2, int idxB1, int idxB2,
                      int code){
    cands.push_back(make_cand_normalized(A1,A2,B1,B2,idxA1,idxA2,idxB1,idxB2,code));
  };

  if (os12 && os34) add_cand(v1,v2, v3,v4, m1.idx,m2.idx, m3.idx,m4.idx, 0);
  if (os13 && os24) add_cand(v1,v3, v2,v4, m1.idx,m3.idx, m2.idx,m4.idx, 1);
  if (os14 && os23) add_cand(v1,v4, v2,v3, m1.idx,m4.idx, m2.idx,m3.idx, 2);

  if (cands.empty()) return false;

  auto best = std::min_element(cands.begin(), cands.end(),
    [&](const Pairing& a, const Pairing& b){
      auto sa = std::make_tuple(std::abs(a.mZa - MZ), std::abs(a.mZb - MZ), -(a.Za.Pt()+a.Zb.Pt()));
      auto sb = std::make_tuple(std::abs(b.mZa - MZ), std::abs(b.mZb - MZ), -(b.Za.Pt()+b.Zb.Pt()));
      return sa < sb;
    });

  bestP = *best;
  sel4 = local; // ordenado por pT desc
  return true;
}

static bool pass_Za_pt(const Pairing& P, const Config& cfg) {
  const double lead = std::max(P.ptZa1, P.ptZa2);
  const double sub  = std::min(P.ptZa1, P.ptZa2);
  return (lead > cfg.PT_LEAD && sub > cfg.PT_SUB);
}

/* ===================== DIAGNÓSTICOS DE PRESELECCIÓN ===================== */
// Cutflow evento (¿quedan >=4 muones tras cada etapa?)
// 1: PT, 2: ETA, 3: ID(Loose), 4: ISO, 5: |dxy|/|dz|, 6: |SIP3D|<4
static TH1I* h_evt_pre_cf = nullptr;

// Distribuciones por muón en cada etapa
static TH1F *h_pt_all=nullptr, *h_pt_pt=nullptr, *h_pt_eta=nullptr, *h_pt_id=nullptr, *h_pt_iso=nullptr, *h_pt_ip=nullptr, *h_pt_sip=nullptr;
static TH1F *h_eta_all=nullptr, *h_eta_pt=nullptr, *h_eta_eta=nullptr, *h_eta_id=nullptr, *h_eta_iso=nullptr, *h_eta_ip=nullptr, *h_eta_sip=nullptr;
static TH1F *h_iso_all=nullptr, *h_iso_id=nullptr;
static TH1F *h_dxy_all=nullptr, *h_dz_all=nullptr, *h_sip_all=nullptr;

static void ensure_diag_hists(){
  if (!h_evt_pre_cf){
    h_evt_pre_cf = new TH1I("h_evt_pre_cf","Event preselect cutflow (>=4 muons);stage;Events",6,0,6);
    h_evt_pre_cf->GetXaxis()->SetBinLabel(1,">=4 (PT)");
    h_evt_pre_cf->GetXaxis()->SetBinLabel(2,">=4 (PT+ETA)");
    h_evt_pre_cf->GetXaxis()->SetBinLabel(3,">=4 (+ID Loose)");
    h_evt_pre_cf->GetXaxis()->SetBinLabel(4,">=4 (+ISO)");
    h_evt_pre_cf->GetXaxis()->SetBinLabel(5,">=4 (+|dxy|,|dz|)");
    h_evt_pre_cf->GetXaxis()->SetBinLabel(6,">=4 (+|SIP3D|<4)");
  }
  auto mk = [&](TH1F*& h, const char* n, const char* t, int nb,double lo,double hi){ if(!h){ h=new TH1F(n,t,nb,lo,hi); h->Sumw2(); } };
  mk(h_pt_all, "h_pt_all",  "muon pT (all);pT [GeV];muons", 120,0,120);
  mk(h_pt_pt,  "h_pt_pt",   "muon pT (after pT);pT [GeV];muons", 120,0,120);
  mk(h_pt_eta, "h_pt_eta",  "muon pT (after |eta|);pT [GeV];muons", 120,0,120);
  mk(h_pt_id,  "h_pt_id",   "muon pT (after ID Loose);pT [GeV];muons", 120,0,120);
  mk(h_pt_iso, "h_pt_iso",  "muon pT (after ISO);pT [GeV];muons", 120,0,120);
  mk(h_pt_ip,  "h_pt_ip",   "muon pT (after |dxy|,|dz|);pT [GeV];muons", 120,0,120);
  mk(h_pt_sip, "h_pt_sip",  "muon pT (after |SIP3D|<4);pT [GeV];muons", 120,0,120);

  mk(h_eta_all,"h_eta_all", "muon |eta| (all);|eta|;muons", 60,0,3.0);
  mk(h_eta_pt, "h_eta_pt",  "muon |eta| (after pT);|eta|;muons", 60,0,3.0);
  mk(h_eta_eta,"h_eta_eta", "muon |eta| (after |eta|);|eta|;muons", 60,0,3.0);
  mk(h_eta_id, "h_eta_id",  "muon |eta| (after ID Loose);|eta|;muons", 60,0,3.0);
  mk(h_eta_iso,"h_eta_iso", "muon |eta| (after ISO);|eta|;muons", 60,0,3.0);
  mk(h_eta_ip, "h_eta_ip",  "muon |eta| (after |dxy|,|dz|);|eta|;muons", 60,0,3.0);
  mk(h_eta_sip,"h_eta_sip", "muon |eta| (after |SIP3D|<4);|eta|;muons", 60,0,3.0);

  mk(h_iso_all,"h_iso_all","muon relISO (all, if exists);relISO;muons",100,0,1.5);
  mk(h_iso_id, "h_iso_id","muon relISO (after ID Loose);relISO;muons",100,0,1.5);

  mk(h_dxy_all,"h_dxy_all","muon |dxy| (all, if exists);|dxy| [cm];muons",100,0,1.0);
  mk(h_dz_all, "h_dz_all", "muon |dz| (all, if exists);|dz| [cm];muons",100,0,5.0);
  mk(h_sip_all,"h_sip_all","muon |SIP3D| (all, if exists);|SIP3D|;muons",100,0,10.0);
}
/* ====================================================================== */

static void run_analysis(const char* inFile, const char* treeName, const char* outFile, const Config& cfg) {
  TH1::AddDirectory(kFALSE);

  TFile fin(inFile, "READ");
  if (fin.IsZombie()) { printf("[ERROR] No pude abrir %s\n", inFile); return; }

  TTree* tr = (TTree*) fin.Get(treeName);
  if (!tr) { printf("[ERROR] No existe TTree '%s'\n", treeName); return; }

  // Activar solo ramas necesarias (si existen)
  tr->SetBranchStatus("*", 0);
  auto enable_if = [&](const char* bname){
    if (tr->GetBranch(bname)) tr->SetBranchStatus(bname, 1);
  };
  enable_if("numbermuon");
  enable_if("muon_pt");   enable_if("muon_eta"); enable_if("muon_phi");
  enable_if("muon_ch");
  enable_if("muon_isLoose"); enable_if("muon_isMedium"); enable_if("muon_isTight");
  enable_if("muon_dxy"); enable_if("muon_dz");
  enable_if("muon_dxyError"); enable_if("muon_dzError");
  enable_if("muon_ip3d"); enable_if("muon_sip3d");
  // ISO ramo forzado (lo activamos explícitamente luego)

  // ---- Branches obligatorias ----
  int numbermuon = 0;
  std::vector<float>* muon_pt  = nullptr;
  std::vector<float>* muon_eta = nullptr;
  std::vector<float>* muon_phi = nullptr;
  std::vector<int>*   muon_ch  = nullptr;

  std::vector<int>*   muon_isLoose  = nullptr;
  std::vector<int>*   muon_isMedium = nullptr;
  std::vector<int>*   muon_isTight  = nullptr;

  tr->SetBranchAddress("numbermuon", &numbermuon);
  tr->SetBranchAddress("muon_pt",    &muon_pt);
  tr->SetBranchAddress("muon_eta",   &muon_eta);
  tr->SetBranchAddress("muon_phi",   &muon_phi);
  tr->SetBranchAddress("muon_ch",    &muon_ch);
  tr->SetBranchAddress("muon_isLoose",  &muon_isLoose);
  tr->SetBranchAddress("muon_isMedium", &muon_isMedium);
  tr->SetBranchAddress("muon_isTight",  &muon_isTight);

  // ---- Branches opcionales usadas en IP/SIP (DECLARACIÓN + CONEXIÓN CONDICIONAL) ----
  std::vector<float>* muon_dxy   = nullptr;
  std::vector<float>* muon_dz    = nullptr;
  std::vector<float>* muon_sip3d = nullptr;

  if (tr->GetBranch("muon_dxy"))    tr->SetBranchAddress("muon_dxy",    &muon_dxy);
  if (tr->GetBranch("muon_dz"))     tr->SetBranchAddress("muon_dz",     &muon_dz);
  if (tr->GetBranch("muon_sip3d"))  tr->SetBranchAddress("muon_sip3d",  &muon_sip3d);

  // ISO opcional: forzar nombre exacto (sin fallback)
  std::vector<float>* muon_iso = nullptr;
  const char* iso_name = nullptr;
  if (cfg.USE_ISO) {
    if (tr->GetBranch(cfg.ISO_BRANCH.c_str())) {
      tr->SetBranchStatus(cfg.ISO_BRANCH.c_str(), 1);
      tr->SetBranchAddress(cfg.ISO_BRANCH.c_str(), &muon_iso);
      iso_name = cfg.ISO_BRANCH.c_str();
      std::cout << "[INFO] Using ISO branch: " << iso_name
                << " (ISO_MAX=" << cfg.ISO_MAX << ")\n";
    } else {
      std::cout << "[WARN] --use-iso pero la rama '" << cfg.ISO_BRANCH
                << "' no existe. No se aplicará ISO.\n";
    }
  }

  // ---- Salida ----
  TFile fout(outFile, "RECREATE");

  // Crear histos de diagnóstico
  ensure_diag_hists();

  // Cutflow (11 pasos definidos)
  g_cutflow = new TH1I("h_cutflow","Cutflow;step;Events",11,0,11);
  auto setlabel=[&](int bin, const char* lab){ g_cutflow->GetXaxis()->SetBinLabel(bin, lab); };
  setlabel(1,"All entries");
  setlabel(2,"core branches ok");
  setlabel(3,"mu preselect (pt/eta/ID/iso/dxy/dz/SIP)");
  setlabel(4,">=4 mu after preselect");
  setlabel(5,"quadruplet built");
  setlabel(6,"OS-OS pairing exists");
  setlabel(7,"Za pT (20/10)");
  setlabel(8,"4mu pT (20/10/7/5) check");
  setlabel(9,"mass windows");
  setlabel(10,"m4mu > 0");
  setlabel(11,"filled");

  // Pares (best/optional pre)
  g_h_mZ12_best = new TH1F("h_mZ12_best","m_{Z(12)} best; m [GeV];Events",120,0,120); g_h_mZ12_best->Sumw2();
  g_h_mZ34_best = new TH1F("h_mZ34_best","m_{Z(34)} best; m [GeV];Events",120,0,120); g_h_mZ34_best->Sumw2();
  g_h_mZ13_best = new TH1F("h_mZ13_best","m_{Z(13)} best; m [GeV];Events",120,0,120); g_h_mZ13_best->Sumw2();
  g_h_mZ24_best = new TH1F("h_mZ24_best","m_{Z(24)} best; m [GeV];Events",120,0,120); g_h_mZ24_best->Sumw2();
  g_h_mZ14_best = new TH1F("h_mZ14_best","m_{Z(14)} best; m [GeV];Events",120,0,120); g_h_mZ14_best->Sumw2();
  g_h_mZ23_best = new TH1F("h_mZ23_best","m_{Z(23)} best; m [GeV];Events",120,0,120); g_h_mZ23_best->Sumw2();

  if (cfg.FILL_PAIR_HIST_PRE) {
    g_h_mZ12_pre = new TH1F("h_mZ12_pre","m_{Z(12)} pre; m [GeV];Events",120,0,120); g_h_mZ12_pre->Sumw2();
    g_h_mZ34_pre = new TH1F("h_mZ34_pre","m_{Z(34)} pre; m [GeV];Events",120,0,120); g_h_mZ34_pre->Sumw2();
    g_h_mZ13_pre = new TH1F("h_mZ13_pre","m_{Z(13)} pre; m [GeV];Events",120,0,120); g_h_mZ13_pre->Sumw2();
    g_h_mZ24_pre = new TH1F("h_mZ24_pre","m_{Z(24)} pre; m [GeV];Events",120,0,120); g_h_mZ24_pre->Sumw2();
    g_h_mZ14_pre = new TH1F("h_mZ14_pre","m_{Z(14)} pre; m [GeV];Events",120,0,120); g_h_mZ14_pre->Sumw2();
    g_h_mZ23_pre = new TH1F("h_mZ23_pre","m_{Z(23)} pre; m [GeV];Events",120,0,120); g_h_mZ23_pre->Sumw2();
  }

  // Finales globales
  g_h_mZa = new TH1F("h_mZa","m_{Z_{a}};m [GeV];Events",120,0,120);     g_h_mZa->Sumw2();
  g_h_mZb = new TH1F("h_mZb","m_{Z_{b}};m [GeV];Events",120,0,120);     g_h_mZb->Sumw2();
  g_h_m4mu= new TH1F("h_m4mu","m_{4#mu};m [GeV];Events",200,0,400);     g_h_m4mu->Sumw2();

  g_h_m4mu_AllTight  = new TH1F("h_m4mu_AllTight","m_{4#mu} AllTight; m [GeV];Events",200,0,400);  g_h_m4mu_AllTight->Sumw2();
  g_h_m4mu_AllMedium = new TH1F("h_m4mu_AllMedium","m_{4#mu} AllMedium; m [GeV];Events",200,0,400); g_h_m4mu_AllMedium->Sumw2();
  g_h_m4mu_AllLoose  = new TH1F("h_m4mu_AllLoose","m_{4#mu} AllLoose; m [GeV];Events",200,0,400);  g_h_m4mu_AllLoose->Sumw2();

  // Árbol de salida
  float mass4mu, pt_4mu, eta_4mu, phi_4mu;
  float px4mu, py4mu, pz4mu, E4mu;
  float mZa_out, mZb_out, dMZa, dMZb;
  int   category; // 3=AllTight, 2=AllMedium, 1=AllLoose, 0=Unclassified
  int   nT, nM, nL;
  int   mu_levels[4]; // niveles de los 4 muones escogidos
  int   idxSel[4];    // índices en el ntuple de los 4 muones finales
  int   idxZa[2], idxZb[2];
  int   pairing_code; // 0=12+34, 1=13+24, 2=14+23

  TTree tout("FourMuTree","Selected 4mu with classification");
  tout.Branch("mass4mu",&mass4mu,"mass4mu/F");
  tout.Branch("pt_4mu",&pt_4mu,"pt_4mu/F");
  tout.Branch("eta_4mu",&eta_4mu,"eta_4mu/F");
  tout.Branch("phi_4mu",&phi_4mu,"phi_4mu/F");
  tout.Branch("px4mu",&px4mu,"px4mu/F");
  tout.Branch("py4mu",&py4mu,"py4mu/F");
  tout.Branch("pz4mu",&pz4mu,"pz4mu/F");
  tout.Branch("E4mu",&E4mu,"E4mu/F");
  tout.Branch("mZa",&mZa_out,"mZa/F");
  tout.Branch("mZb",&mZb_out,"mZb/F");
  tout.Branch("dMZa",&dMZa,"dMZa/F");
  tout.Branch("dMZb",&dMZb,"dMZb/F");
  tout.Branch("category",&category,"category/I");
  tout.Branch("nT",&nT,"nT/I");
  tout.Branch("nM",&nM,"nM/I");
  tout.Branch("nL",&nL,"nL/I");
  tout.Branch("mu_levels",mu_levels,"mu_levels[4]/I");
  tout.Branch("idxSel",idxSel,"idxSel[4]/I");
  tout.Branch("idxZa",idxZa,"idxZa[2]/I");
  tout.Branch("idxZb",idxZb,"idxZb[2]/I");
  tout.Branch("pairing_code",&pairing_code,"pairing_code/I");

  const Long64_t nent = tr->GetEntries();
  for (Long64_t ie=0; ie<nent; ++ie) {
    g_cutflow->Fill(1);
    tr->GetEntry(ie);

    if (!muon_pt || !muon_eta || !muon_phi || !muon_ch ||
        !muon_isLoose || !muon_isMedium || !muon_isTight) continue;
    g_cutflow->Fill(2);

    /* ===================== Preselección instrumentada ===================== */
    std::vector<int> idx_all, idx_pt, idx_eta, idx_id, idx_iso, idx_ip, idx_sip;

    auto exists = [&](auto* v, int i){ return v && ((int)v->size()>i); };
    auto isLooseF = [&](int i)->bool{
      if (!muon_isLoose) return true;
      if (!exists(muon_isLoose,i)) return false;
      return (*muon_isLoose)[i]!=0;
    };
    auto passISO = [&](int i)->bool{
      if (!cfg.USE_ISO || !muon_iso) return true;
      if (!exists(muon_iso,i)) return true;             // si falta, NO cortes por ISO
      float iso = (*muon_iso)[i];
      if (!std::isfinite(iso)) return true;             // NaN/Inf => NO cortes
      return iso <= (float)cfg.ISO_MAX;
    };
    auto passIP = [&](int i)->bool{
      if (!cfg.APPLY_IP) return true;
      if (exists(muon_dxy,i) && std::abs((*muon_dxy)[i]) >= cfg.DXY_MAX) return false;
      if (exists(muon_dz ,i) && std::abs((*muon_dz )[i]) >= cfg.DZ_MAX ) return false;
      return true;
    };
    auto passSIP = [&](int i)->bool{
      if (!cfg.APPLY_SIP) return true;
      if (muon_sip3d && exists(muon_sip3d,i)) return (std::abs((*muon_sip3d)[i]) < (float)cfg.SIP3D_MAX);
      return true; // si no hay SIP3D, no cortar por SIP
    };

    // Llenar "all" (referencias)
    if (muon_pt && muon_eta){
      const int n = (int)muon_pt->size();
      for (int i=0;i<n;++i){
        idx_all.push_back(i);
        h_pt_all->Fill( (*muon_pt)[i] );
        h_eta_all->Fill( std::abs((*muon_eta)[i]) );
        if (muon_iso  && exists(muon_iso,i))   h_iso_all->Fill((*muon_iso)[i]);
        if (exists(muon_dxy,i))   h_dxy_all->Fill(std::abs((*muon_dxy)[i]));
        if (exists(muon_dz,i))    h_dz_all->Fill( std::abs((*muon_dz)[i]));
        if (muon_sip3d&& exists(muon_sip3d,i)) h_sip_all->Fill(std::abs((*muon_sip3d)[i]));
      }
    }

    // PT
    for (int i: idx_all){
      if (!exists(muon_pt,i)) continue;
      if ((*muon_pt)[i] <= cfg.PT_MIN_BASE) continue;
      idx_pt.push_back(i);
      h_pt_pt->Fill((*muon_pt)[i]);
      h_eta_pt->Fill(std::abs((*muon_eta)[i]));
    }
    // |ETA|
    for (int i: idx_pt){
      if (cfg.APPLY_ETA && std::abs((*muon_eta)[i]) > cfg.ETA_MAX) continue;
      idx_eta.push_back(i);
      h_pt_eta->Fill((*muon_pt)[i]);
      h_eta_eta->Fill(std::abs((*muon_eta)[i]));
    }
    // ID (≥Loose)
    for (int i: idx_eta){
      if (cfg.APPLY_ID && !isLooseF(i)) continue;
      idx_id.push_back(i);
      h_pt_id->Fill((*muon_pt)[i]);
      h_eta_id->Fill(std::abs((*muon_eta)[i]));
      if (muon_iso && exists(muon_iso,i)) h_iso_id->Fill((*muon_iso)[i]);
    }
    // ISO (si procede)
    if (cfg.USE_ISO && muon_iso && !(!cfg.USE_ISO)) {
      for (int i: idx_id){
        if (!passISO(i)) continue;
        idx_iso.push_back(i);
        h_pt_iso->Fill((*muon_pt)[i]);
        h_eta_iso->Fill(std::abs((*muon_eta)[i]));
      }
    } else {
      idx_iso = idx_id;
    }
    // |dxy|,|dz|
    for (int i: idx_iso){
      if (!passIP(i)) continue;
      idx_ip.push_back(i);
      h_pt_ip->Fill((*muon_pt)[i]);
      h_eta_ip->Fill(std::abs((*muon_eta)[i]));
    }
    // SIP3D
    for (int i: idx_ip){
      if (!passSIP(i)) continue;
      idx_sip.push_back(i);
      h_pt_sip->Fill((*muon_pt)[i]);
      h_eta_sip->Fill(std::abs((*muon_eta)[i]));
    }

    // Cutflow evento: ¿quedan ≥4 muones tras cada etapa?
    if ((int)idx_pt.size()  >=4) h_evt_pre_cf->Fill(1);
    if ((int)idx_eta.size() >=4) h_evt_pre_cf->Fill(2);
    if ((int)idx_id.size()  >=4) h_evt_pre_cf->Fill(3);
    if ((int)idx_iso.size() >=4) h_evt_pre_cf->Fill(4);
    if ((int)idx_ip.size()  >=4) h_evt_pre_cf->Fill(5);
    if ((int)idx_sip.size() >=4) h_evt_pre_cf->Fill(6);

    // Construir 'mus' desde los que sobrevivieron TODAS las etapas
    std::vector<Mu> mus; mus.reserve(idx_sip.size());
    for (int i: idx_sip){
      const double pt  = (*muon_pt)[i];
      const double eta = (*muon_eta)[i];
      const double phi = (*muon_phi)[i];
      const int    q   = (*muon_ch)[i];
      int lvl = muLevel((*muon_isLoose)[i], (*muon_isMedium)[i], (*muon_isTight)[i]);
      mus.push_back(Mu{pt, eta, phi, q, lvl, i});
    }
    /* =================== Fin de preselección instrumentada =================== */

    // Paso 3/4 del cutflow general
    g_cutflow->Fill(3);
    if ((int)mus.size() < 4) continue;
    g_cutflow->Fill(4);

    // -------- Construcción del cuádruple y pairing --------
    std::vector<Mu> sel; sel.reserve(4);
    Pairing best{}; bool have_pairing = false;

    auto try_best_for_top4 = [&](){
      std::sort(mus.begin(), mus.end(), [](const Mu& a, const Mu& b){return a.pt>b.pt;});
      sel = {mus[0], mus[1], mus[2], mus[3]};
      if ( (sel[0].q + sel[1].q + sel[2].q + sel[3].q) != 0 ) return false; // carga total 0
      std::array<Mu,4> m4 = {sel[0], sel[1], sel[2], sel[3]};
      std::vector<Mu> sel4;
      if (!best_pairing_for_four(m4, best, sel4)) return false;
      sel = sel4; // ordenado por pT
      return true;
    };

    if (cfg.USE_COMB_SEARCH) {
      const int N = (int)mus.size();
      double bestKeyA=1e9, bestKeyB=1e9; double bestKeyC=-1e9;
      for (int a=0;a<N-3;++a)
        for (int b=a+1;b<N-2;++b)
          for (int c=b+1;c<N-1;++c)
            for (int d=c+1; d<N; ++d) {
              std::array<Mu,4> m4 = {mus[a], mus[b], mus[c], mus[d]};
              // (opcional) podrías filtrar por carga total==0 aquí para acelerar
              Pairing trial; std::vector<Mu> sel4;
              if (!best_pairing_for_four(m4, trial, sel4)) continue;
              double keyA = std::abs(trial.mZa - MZ);
              double keyB = std::abs(trial.mZb - MZ);
              double keyC = (trial.Za.Pt()+trial.Zb.Pt());
              bool better = (keyA<bestKeyA) || (keyA==bestKeyA && (keyB<bestKeyB || (keyB==bestKeyB && keyC>bestKeyC)));
              if (better) {
                best = trial; sel = sel4;
                bestKeyA=keyA; bestKeyB=keyB; bestKeyC=keyC;
                have_pairing = true;
              }
            }
      if (!have_pairing) continue;
      g_cutflow->Fill(5);
      g_cutflow->Fill(6);
    } else {
      if (!try_best_for_top4()) continue;
      g_cutflow->Fill(5);
      g_cutflow->Fill(6);
    }

    // Corte pT en Za (20/10)
    if (!pass_Za_pt(best, cfg)) continue;
    g_cutflow->Fill(7);

    // Corte 20/10/7/5 en los 4 muones (si se pide)
    if (cfg.ENFORCE_4MU_PT) {
      std::array<double,4> pts = { sel[0].pt, sel[1].pt, sel[2].pt, sel[3].pt };
      std::sort(pts.begin(), pts.end(), std::greater<double>());
      if (!pass_4mu_pt(pts, cfg)) continue;
    }
    g_cutflow->Fill(8);

    // Ventanas de masa Za y Zb (configurables)
    if (!(best.mZa > cfg.MZA_MIN && best.mZa < cfg.MZA_MAX &&
          best.mZb > cfg.MZB_MIN && best.mZb < cfg.MZB_MAX)) continue;
    g_cutflow->Fill(9);

    // ---- Sistema 4µ ----
    TLorentzVector H = best.Za + best.Zb;
    mass4mu = H.M();
    if (mass4mu <= 0.) continue;
    g_cutflow->Fill(10);

    // Kinemática 4µ
    pt_4mu  = H.Pt();
    eta_4mu = H.Eta();
    phi_4mu = H.Phi();
    px4mu   = H.Px();
    py4mu   = H.Py();
    pz4mu   = H.Pz();
    E4mu    = H.E();

    // Guardar pares y trazas
    mZa_out = best.mZa;
    mZb_out = best.mZb;
    dMZa = std::abs(best.mZa - MZ);
    dMZb = std::abs(best.mZb - MZ);
    pairing_code = best.pairing_code;

    // Clasificación y combinación
    int cat = classify_event(sel, nT, nM, nL); category = cat;
    for (int k=0;k<4;++k) { mu_levels[k] = sel[k].level; idxSel[k] = sel[k].idx; }
    idxZa[0]=best.idxZa1; idxZa[1]=best.idxZa2;
    idxZb[0]=best.idxZb1; idxZb[1]=best.idxZb2;

    // (1) Pares del pareo ganador: SOLO las dos parejas elegidas (Za y Zb)
    if (cfg.FILL_PAIR_HIST_BEST_ONLY) {
      TLorentzVector v1=p4_of(sel[0]), v2=p4_of(sel[1]), v3=p4_of(sel[2]), v4=p4_of(sel[3]);
      auto fill_pair = [&](int i, int j){
        const TLorentzVector* vv[4] = { &v1, &v2, &v3, &v4 };
        if      ((i==0&&j==1)||(i==1&&j==0)) { if (g_h_mZ12_best) g_h_mZ12_best->Fill( (*vv[i]+*vv[j]).M() ); }
        else if ((i==2&&j==3)||(i==3&&j==2)) { if (g_h_mZ34_best) g_h_mZ34_best->Fill( (*vv[i]+*vv[j]).M() ); }
        else if ((i==0&&j==2)||(i==2&&j==0)) { if (g_h_mZ13_best) g_h_mZ13_best->Fill( (*vv[i]+*vv[j]).M() ); }
        else if ((i==1&&j==3)||(i==3&&j==1)) { if (g_h_mZ24_best) g_h_mZ24_best->Fill( (*vv[i]+*vv[j]).M() ); }
        else if ((i==0&&j==3)||(i==3&&j==0)) { if (g_h_mZ14_best) g_h_mZ14_best->Fill( (*vv[i]+*vv[j]).M() ); }
        else if ((i==1&&j==2)||(i==2&&j==1)) { if (g_h_mZ23_best) g_h_mZ23_best->Fill( (*vv[i]+*vv[j]).M() ); }
      };
      auto find_pos = [&](int idx)->int{
        for (int k=0;k<4;++k) if (sel[k].idx == idx) return k;
        return -1;
      };
      int za_i = find_pos(best.idxZa1), za_j = find_pos(best.idxZa2);
      int zb_i = find_pos(best.idxZb1), zb_j = find_pos(best.idxZb2);
      if (za_i>=0 && za_j>=0) fill_pair(za_i, za_j);
      if (zb_i>=0 && zb_j>=0) fill_pair(zb_i, zb_j);
    }

    // (2) Za/Zb y m4l global
    g_h_mZa->Fill(mZa_out);
    g_h_mZb->Fill(mZb_out);
    g_h_m4mu->Fill(mass4mu);

    // (3) Tradicionales
    if (category==3)      g_h_m4mu_AllTight->Fill(mass4mu);
    else if (category==2) g_h_m4mu_AllMedium->Fill(mass4mu);
    else if (category==1) g_h_m4mu_AllLoose->Fill(mass4mu);

    // (4) Combinaciones T{nT}M{nM}L{nL}
    {
      const std::string key = comp_key(nT,nM,nL);
      TH1F* h4  = ensure_hist(g_h_m4mu_by_comp, key, "m4mu_by_comp", 200, 0, 400);
      TH1F* hZa = ensure_hist(g_h_mZa_by_comp,  key, "mZa_by_comp",  120, 0, 120);
      TH1F* hZb = ensure_hist(g_h_mZb_by_comp,  key, "mZb_by_comp",  120, 0, 120);
      h4->Fill(mass4mu);
      hZa->Fill(mZa_out);
      hZb->Fill(mZb_out);
    }

    tout.Fill();
    g_cutflow->Fill(11);
  }

  // Resumen cutflow
  std::cout << "\n=== Cutflow resumen ===\n";
  for (int b=1; b<=11; ++b) {
    const char* lab = g_cutflow->GetXaxis()->GetBinLabel(b);
    std::cout << b << " : " << (lab?lab:"") << " -> " << g_cutflow->GetBinContent(b) << "\n";
  }
  std::cout << "Salida en " << outFile << std::endl;

  // Guardar todo
  fout.cd();
  g_cutflow->Write();

  if (g_h_mZ12_pre) { g_h_mZ12_pre->Write(); g_h_mZ34_pre->Write();
                      g_h_mZ13_pre->Write(); g_h_mZ24_pre->Write();
                      g_h_mZ14_pre->Write(); g_h_mZ23_pre->Write(); }

  g_h_mZ12_best->Write(); g_h_mZ34_best->Write();
  g_h_mZ13_best->Write(); g_h_mZ24_best->Write();
  g_h_mZ14_best->Write(); g_h_mZ23_best->Write();

  g_h_mZa->Write();  g_h_mZb->Write();
  g_h_m4mu->Write();
  g_h_m4mu_AllTight->Write();
  g_h_m4mu_AllMedium->Write();
  g_h_m4mu_AllLoose->Write();

  for (auto& kv : g_h_m4mu_by_comp) kv.second->Write();
  for (auto& kv : g_h_mZa_by_comp)  kv.second->Write();
  for (auto& kv : g_h_mZb_by_comp)  kv.second->Write();

  if (h_evt_pre_cf) h_evt_pre_cf->Write();
  if (h_pt_all){
    h_pt_all->Write();  h_pt_pt->Write();  h_pt_eta->Write();  h_pt_id->Write();
    if (h_pt_iso) h_pt_iso->Write();
    h_pt_ip->Write();   h_pt_sip->Write();
  }
  if (h_eta_all){
    h_eta_all->Write(); h_eta_pt->Write(); h_eta_eta->Write(); h_eta_id->Write();
    if (h_eta_iso) h_eta_iso->Write();
    h_eta_ip->Write();  h_eta_sip->Write();
  }
  if (h_iso_all) h_iso_all->Write();
  if (h_iso_id)  h_iso_id->Write();
  if (h_dxy_all) h_dxy_all->Write();
  if (h_dz_all)  h_dz_all->Write();
  if (h_sip_all) h_sip_all->Write();

  tout.Write();
  fout.Close();
  fin.Close();
}

// ===== CLI sencillo para toggles =====
static void parse_cli(Config& cfg, int argc, char** argv) {
  auto getd = [&](const char* pre, double& var){
    for (int i=1;i<argc;++i){
      std::string s = argv[i];
      if (s.rfind(pre,0)==0) { var = atof(s.substr(std::string(pre).size()).c_str()); }
    }
  };
  auto gets = [&](const char* pre, std::string& var){
    for (int i=1;i<argc;++i){
      std::string s = argv[i];
      if (s.rfind(pre,0)==0) { var = s.substr(std::string(pre).size()); }
    }
  };
  auto has = [&](const char* flag){
    for (int i=1;i<argc;++i) if (std::string(argv[i])==flag) return true;
    return false;
  };

  // numéricos (básicos)
  getd("--eta-max=",   cfg.ETA_MAX);
  getd("--pt-min=",    cfg.PT_MIN_BASE);
  getd("--dxy-max=",   cfg.DXY_MAX);
  getd("--dz-max=",    cfg.DZ_MAX);
  getd("--sip3d-max=", cfg.SIP3D_MAX);

  // ISO
  getd("--iso-max=",   cfg.ISO_MAX);
  gets("--iso-branch=", cfg.ISO_BRANCH);

  // pT requeridos
  getd("--pt-lead=",   cfg.PT_LEAD);
  getd("--pt-sub=",    cfg.PT_SUB);
  getd("--pt-third=",  cfg.PT_THIRD);
  getd("--pt-fourth=", cfg.PT_FOURTH);

  // Ventanas de masa
  getd("--mza-min=",   cfg.MZA_MIN);
  getd("--mza-max=",   cfg.MZA_MAX);
  getd("--mzb-min=",   cfg.MZB_MIN);
  getd("--mzb-max=",   cfg.MZB_MAX);

  // booleanos (flags)
  cfg.USE_ISO             = has("--use-iso");
  cfg.ENFORCE_4MU_PT      = has("--enforce-4mu-pt");
  cfg.USE_COMB_SEARCH     = has("--comb-search");
  cfg.FILL_PAIR_HIST_PRE  = has("--fill-pairs-pre");
  if (has("--no-fill-pairs-best")) cfg.FILL_PAIR_HIST_BEST_ONLY = false;

  // desactivar etapas
  if (has("--no-eta")) cfg.APPLY_ETA=false;
  if (has("--no-id"))  cfg.APPLY_ID=false;
  if (has("--no-iso")) cfg.USE_ISO=false;
  if (has("--no-ip"))  cfg.APPLY_IP=false;
  if (has("--no-sip")) cfg.APPLY_SIP=false;
}

// ===== main =====
int main(int argc, char** argv){
  // Posicionales por defecto
  const char* in   = "analisis_final_clasificados.root";
  const char* tree = "Events";
  const char* out  = "out_zz4mu.root";

  // Permite sobreescribir posicionales si vienen primero sin '-'
  int p = 1;
  if (p<argc && argv[p][0] != '-') in = argv[p++];
  if (p<argc && argv[p][0] != '-') tree = argv[p++];
  if (p<argc && argv[p][0] != '-') out = argv[p++];

  Config cfg;
  parse_cli(cfg, argc, argv);

  std::cout << "Config:\n"
            << "  Inputs: file=" << in << " tree=" << tree << " out=" << out << "\n"
            << "  Muon cuts: pt_min>" << cfg.PT_MIN_BASE
            << ", |eta|<" << (cfg.APPLY_ETA?std::to_string(cfg.ETA_MAX):std::string("DISABLED"))
            << ", ID>=" << (cfg.APPLY_ID?"Loose":"DISABLED")
            << ", |dxy|<" << (cfg.APPLY_IP?std::to_string(cfg.DXY_MAX):std::string("DISABLED"))
            << ", |dz|<"  << (cfg.APPLY_IP?std::to_string(cfg.DZ_MAX):std::string("DISABLED"))
            << ", |SIP3D|<" << (cfg.APPLY_SIP?std::to_string(cfg.SIP3D_MAX):std::string("DISABLED"))
            << "\n";
  if (cfg.USE_ISO) {
    std::cout << "  ISO: use_iso=true, branch=" << cfg.ISO_BRANCH
              << ", iso_max=" << cfg.ISO_MAX << "\n";
  } else {
    std::cout << "  ISO: use_iso=false\n";
  }
  std::cout << "  Za pT: lead>" << cfg.PT_LEAD << ", sub>" << cfg.PT_SUB
            << " | 4mu pT check=" << (cfg.ENFORCE_4MU_PT?"ON":"OFF")
            << " (third>" << cfg.PT_THIRD << ", fourth>" << cfg.PT_FOURTH << ")\n"
            << "  Mass windows: Za[" << cfg.MZA_MIN << "," << cfg.MZA_MAX << "], "
            << "Zb[" << cfg.MZB_MIN << "," << cfg.MZB_MAX << "]\n"
            << "  comb_search=" << (cfg.USE_COMB_SEARCH?"true":"false")
            << "\n";

  run_analysis(in, tree, out, cfg);
  return 0;
}
