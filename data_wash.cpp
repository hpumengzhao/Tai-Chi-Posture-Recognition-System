/*
* @author:  codancer
* @createTime:  2021-04-22, 16:57:12
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <ctime>
#include <cassert>
#include <complex>
#include <string>
#include <cstring>
#include <chrono>
#include <random>
#include <bitset>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;
const ll mod = 1e9+7;
#define pb push_back
#define fi first
#define se second
#define SZ(x) ((int)(x).size())
#define rep(i,a,b) for(int i=(a);i<=(b);i++)
#define fep(i,a,b) for(int i=(a);i>=(b);i--)
#define deb(x) cerr<<#x<<" = "<<(x)<<"\n"
typedef vector<int> VI;
typedef vector<ll> VII;
typedef pair<int,int> pii;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
ll Rand(ll B) {
	return (ull)rng() % B;
}
//data[i][j][k][3]:姿势i的第j步的第k个关节
int main(){
	int n;
	cout<<'[';
	rep(t,1,17){
		cin>>n;
		string s;
		cout<<'[';
		rep(i,1,n+1){
			getline(cin,s);
			if(i==1) continue;
			cout<<'[';
			string t="";
			for(int j=0;j<(int)s.length()-1;j++){
				if(s[j]==':') s[j]=',';
			}
			for(int j=0;j<(int)s.length()-2;j++){
				if(s[j]=='('||s[j]==')') continue;
				t+=s[j];
				if(s[j]==']') t+=",";
			}
			t+=']';
			cout<<t;
			if(i==n+1) cout<<']';
			else cout<<"],";
			cout<<"\\"<<endl;
		}
		if(t==17) cout<<']';
		else cout<<"],";
	}
	cout<<']';
	return 0;
}