#include "raw_stream_moments.hpp"
#include <vector>
#include <fstream>
#include <iostream>
template<class T>static void put(const std::string&p,const std::vector<T>&v){std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(T));}
int main(int argc,char**argv){
 if(argc!=2)return 2;const std::string out=argv[1];const int count=8192;std::vector<float>x(count),y(count);std::vector<uint8_t>tags(count);std::vector<double>w(count);
 for(int i=0;i<count;i++){x[i]=(float)(1e8+1e4*std::sin(.137*i)+500*std::cos(.003*i));y[i]=(float)(2.3*x[i]+1700+1200*std::cos(.927*i));tags[i]=i%3;w[i]=(i%5)?(double)(1+i%9):0;if(w[i]==0){x[i]=0;y[i]=0;}}
 put(out+"/x.f32",x);put(out+"/y.f32",y);put(out+"/tags.u8",tags);put(out+"/w.f64",w);RawPairMoments m[2][4];
 for(int off=0;off<count;off+=137){int take=std::min(137,count-off);for(int domain=0;domain<2;domain++)for(int band=-1;band<3;band++)m[domain][band+1].add_block(x.data()+off,y.data()+off,tags.data()+off,w.data()+off,take,band,domain==1);}
 long double direct[2][4]={};for(int domain=0;domain<2;domain++){auto fit=m[domain][0].affine();for(int i=0;i<count;i++){if(domain&&w[i]<=0)continue;long double e=fit.first*(long double)y[i]+fit.second-(long double)x[i];direct[domain][0]+=e*e;direct[domain][tags[i]+1]+=e*e;}}
 std::cout<<"{";for(int domain=0;domain<2;domain++){if(domain)std::cout<<",";auto fit=m[domain][0].affine();std::cout<<"\""<<(domain?"valid":"all")<<"\":{";for(int band=0;band<4;band++){if(band)std::cout<<",";std::cout<<"\""<<band<<"\":"<<m[domain][band].json(fit.first,fit.second,&direct[domain][band]);}std::cout<<"}";}std::cout<<"}\n";
}
