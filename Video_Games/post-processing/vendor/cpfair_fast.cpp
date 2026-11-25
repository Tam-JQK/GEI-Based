#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
struct Entry { double v; int id; };
extern "C" int cpfair(const float* scores,const int32_t* groups,const int32_t* excluded_offsets,const int32_t* excluded,int users,int items,int ngrp,double lambda,int32_t* output){
 std::vector<double> counts(ngrp,0),bonus(ngrp); std::vector<Entry> buffer; buffer.reserve(items); std::vector<int> marks(items,-1); double total=0;
 for(int u=0;u<users;u++){
  for(int j=excluded_offsets[u];j<excluded_offsets[u+1];j++) marks[excluded[j]]=u;
  for(int g=0;g<ngrp;g++)bonus[g]=lambda*(1-counts[g]/(total+1e-5));
  buffer.clear();for(int i=0;i<items;i++)if(marks[i]!=u)buffer.push_back({double(scores[size_t(u)*items+i])+bonus[groups[i]],i});
  if(buffer.size()<20)return 1;
  auto cmp=[](const Entry&a,const Entry&b){return a.v>b.v||(a.v==b.v&&a.id>b.id);};
  std::partial_sort(buffer.begin(),buffer.begin()+20,buffer.end(),cmp);
  for(int k=0;k<20;k++){int id=buffer[k].id;output[size_t(u)*20+k]=id;counts[groups[id]]++;}total+=20;
 }return 0;
}
