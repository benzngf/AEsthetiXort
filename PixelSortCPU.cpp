#include "PixelSort_AE.h"
#include <math.h>
AEGP_SuiteHandler *gSuites;
float CalcKeyCPUHue(float minThres, float maxThres, Pixel &p) {
	if (p.a <= 0.0f || minThres == maxThres) {
		p.key = -1.0f;
		return	-1.0f;
	}
	float minV = (p.e[0] > p.e[1]) ? ((p.e[1] > p.e[2]) ? p.e[2] : p.e[1]) : ((p.e[0] > p.e[2]) ? p.e[2] : p.e[0]);
	if (p.e[0] > p.e[1] && p.e[0] > p.e[1]) {//R max
		p.key = (p.g - p.b) / (p.r - minV);
	}
	else if (p.e[1] > p.e[0] && p.e[1] > p.e[2]) {//G max
		p.key = 2.0f+(p.b - p.r) / (p.g - minV);
	}else {//B max
		p.key =4.0f+ (p.r - p.g) / (p.b - minV);
	}
	if (p.key < 0) p.key += 6.0f;
	p.key = p.key*100.0f / 6.0f;
	if (minThres < maxThres) {
		if (p.key > maxThres) {
			p.key = -1.0f;
		}
		else {
			p.key -= minThres;
		}
	}
	else {
		if (p.key < maxThres) {
			p.key +=(100.0f-minThres);
		}else {
			p.key -= minThres;
		}
	}
	return	p.key;
}
float CalcKeyCPUSat(float minThres, float maxThres, Pixel &p) {
	if (p.a <= 0.0f || minThres == maxThres) {
		p.key = -1.0f;
		return	-1.0f;
	}
	float minV = (p.e[0] > p.e[1]) ? ((p.e[1] > p.e[2]) ? p.e[2] : p.e[1]) : ((p.e[0] > p.e[2]) ? p.e[2] : p.e[0]);
	float maxV = (p.e[0] < p.e[1]) ? ((p.e[1] < p.e[2]) ? p.e[2] : p.e[1]) : ((p.e[0] < p.e[2]) ? p.e[2] : p.e[0]);
	float l = (maxV + minV) / 2;
	if (maxV == minV) {
		p.key = 0.0f;
	}
	else {
		p.key = (l > 127.5f) ? ((maxV - minV) / (510.0f - maxV - minV)) : ((maxV - minV) / (maxV + minV));
	}
	p.key = p.key*100.0f;
	if (minThres < maxThres) {
		if (p.key > maxThres) {
			p.key = -1.0f;
		}
		else {
			p.key -= minThres;
		}
	}
	else {
		if (p.key < maxThres) {
			p.key += (100.0f - minThres);
		}
		else {
			p.key -= minThres;
		}
	}
	return	p.key;
}
float CalcKeyCPULum(float minThres, float maxThres, Pixel &p) {
	if (p.a <= 0.0f || minThres == maxThres) {
		p.key = -1.0f;
		return	-1.0f;
	}
	float minV = (p.e[0] > p.e[1]) ? ((p.e[1] > p.e[2]) ? p.e[2] : p.e[1]) : ((p.e[0] > p.e[2]) ? p.e[2] : p.e[0]);
	float maxV = (p.e[0] < p.e[1]) ? ((p.e[1] < p.e[2]) ? p.e[2] : p.e[1]) : ((p.e[0] < p.e[2]) ? p.e[2] : p.e[0]);
	p.key = (maxV + minV)*10.0f / 51.0f;
	if (minThres < maxThres) {
		if (p.key > maxThres) {
			p.key = -1.0f;
		}
		else {
			p.key -= minThres;
		}
	}
	else {
		if (p.key < maxThres) {
			p.key += (100.0f - minThres);
		}
		else {
			p.key -= minThres;
		}
	}
	return	p.key;
}
float CalcKeyCPUR(float minThres, float maxThres, Pixel &p) {
	if (p.a <= 0.0f || minThres == maxThres) {
		p.key = -1.0f;
		return	-1.0f;
	}
	p.key = p.r*20.0f / 51.0f;
	if (minThres < maxThres) {
		if (p.key > maxThres) {
			p.key = -1.0f;
		}
		else {
			p.key -= minThres;
		}
	}
	else {
		if (p.key < maxThres) {
			p.key += (100.0f - minThres);
		}
		else {
			p.key -= minThres;
		}
	}
	return	p.key;
}
float CalcKeyCPUG(float minThres, float maxThres, Pixel &p) {
	if (p.a <= 0.0f || minThres == maxThres) {
		p.key = -1.0f;
		return	-1.0f;
	}
	p.key = p.g*20.0f / 51.0f;
	if (minThres < maxThres) {
		if (p.key > maxThres) {
			p.key = -1.0f;
		}
		else {
			p.key -= minThres;
		}
	}
	else {
		if (p.key < maxThres) {
			p.key += (100.0f - minThres);
		}
		else {
			p.key -= minThres;
		}
	}
	return	p.key;
}
float CalcKeyCPUB(float minThres, float maxThres, Pixel &p) {
	if (p.a <= 0.0f || minThres== maxThres) {
		p.key = -1.0f;
		return	-1.0f;
	}
	p.key = p.b*20.0f / 51.0f;
	if (minThres < maxThres) {
		if (p.key > maxThres) {
			p.key = -1.0f;
		}
		else {
			p.key -= minThres;
		}
	}
	else {
		if (p.key < maxThres) {
			p.key += (100.0f - minThres);
		}
		else {
			p.key -= minThres;
		}
	}
	return	p.key;
}

typedef struct pPoint {
	float x;
	float y;
} PPoint;

void Bilinear(PPoint pos, Pixel* P00, Pixel* P01, Pixel* P10, Pixel* P11, Pixel* Pout) {
	Pout->r = P00->r*(1.0f - pos.x)*(1.0f - pos.y) + P10->r*pos.x*(1.0f - pos.y) + P01->r*(1.0f - pos.x)*pos.y + P11->r*pos.x*pos.y;
	Pout->g = P00->g*(1.0f - pos.x)*(1.0f - pos.y) + P10->g*pos.x*(1.0f - pos.y) + P01->g*(1.0f - pos.x)*pos.y + P11->g*pos.x*pos.y;
	Pout->b = P00->b*(1.0f - pos.x)*(1.0f - pos.y) + P10->b*pos.x*(1.0f - pos.y) + P01->b*(1.0f - pos.x)*pos.y + P11->b*pos.x*pos.y;
	Pout->a = P00->a*(1.0f - pos.x)*(1.0f - pos.y) + P10->a*pos.x*(1.0f - pos.y) + P01->a*(1.0f - pos.x)*pos.y + P11->a*pos.x*pos.y;
	Pout->key = P00->key*(1.0f - pos.x)*(1.0f - pos.y) + P10->key*pos.x*(1.0f - pos.y) + P01->key*(1.0f - pos.x)*pos.y + P11->key*pos.x*pos.y;
}

PPoint traverseLinear(int w, int h, PPoint now, Pixel* footage, bool previous, bool biInter,PixelSortPatternParm* param, Pixel* output) {
	if (now.x < 0) return now;
	PixelSortPatternParmLinear* Lparam = (PixelSortPatternParmLinear*)param;
	PPoint pRet;
	if (previous) {
		pRet.x = now.x - (float)gSuites->ANSICallbacksSuite1()->cos(Lparam->angle);
		pRet.y = now.y - (float)gSuites->ANSICallbacksSuite1()->sin(Lparam->angle);
	}
	else {
		pRet.x = now.x + (float)gSuites->ANSICallbacksSuite1()->cos(Lparam->angle);
		pRet.y = now.y + (float)gSuites->ANSICallbacksSuite1()->sin(Lparam->angle);
	}
	if (pRet.x<0 || pRet.x>=w || pRet.y<0 || pRet.y>=h || footage[((int)pRet.y)*w + (int)pRet.x].key < 0) {
		pRet.x = -100.0f;
		pRet.y = -100.0f;
		return pRet;
	}
	if (output != nullptr) {
		if (biInter) {
			PPoint pBi;
			pBi.x = pRet.x - ((int)pRet.x);
			pBi.y = pRet.y - ((int)pRet.y);
			int xside = pRet.x >= w - 1;
			int yside = pRet.y >= h - 1;
			Bilinear(pBi, &footage[((int)pRet.y)*w + (int)pRet.x], &footage[((int)pRet.y + 1-yside)*w + (int)pRet.x], &footage[((int)pRet.y)*w + (int)pRet.x + 1-xside], &footage[((int)pRet.y + 1-yside)*w + (int)pRet.x + 1-xside], output);
		}else {
			output->r = footage[((int)pRet.y)*w + (int)pRet.x].r;
			output->g = footage[((int)pRet.y)*w + (int)pRet.x].g;
			output->b = footage[((int)pRet.y)*w + (int)pRet.x].b;
			output->a = footage[((int)pRet.y)*w + (int)pRet.x].a;
			output->key = footage[((int)pRet.y)*w + (int)pRet.x].key;
		}
	}
	return pRet;
}
PPoint traverseSpiral(int w, int h, PPoint now, Pixel* footage, bool previous, bool biInter, PixelSortPatternParm* param, Pixel* output) {
	if (now.x < 0) return now;
	PixelSortPatternParmSpiral* Sparam = (PixelSortPatternParmSpiral*)param;
	PPoint pRet;
	float r = sqrtf((now.x - Sparam->center[0])*(now.x - Sparam->center[0]) + (now.y - Sparam->center[1])*(now.y - Sparam->center[1]));
	float theta = atan2f(now.y - Sparam->center[1], now.x - Sparam->center[0]);
	float dr = 100.0f / (100.0f + (r*Sparam->curveAngle/100.0f)*(r*Sparam->curveAngle/100.0f));
	float dTheta = dr*Sparam->curveAngle / 1000.0f;
	if (previous) {
		if((int)now.x == (int)Sparam->center[0] && (int)now.y == (int)Sparam->center[1]) {
			pRet.x = -100.0f;
			pRet.y = -100.0f;
			return pRet;
		}
		else {
			r -= dr;
			theta -= dTheta;
			if (r <= 0) {
				pRet.x = Sparam->center[0];
				pRet.y = Sparam->center[1];
			}
			else {
				pRet.x = Sparam->center[0] + r*cosf(theta);
				pRet.y = Sparam->center[1] + r*sinf(theta);
			}
		}
	}
	else {
		r += dr;
		theta += dTheta;
		pRet.x = Sparam->center[0] + r*cosf(theta);
		pRet.y = Sparam->center[1] + r*sinf(theta);
	}
	if (pRet.x<0 || pRet.x >= w || pRet.y<0 || pRet.y >= h || footage[((int)pRet.y)*w + (int)pRet.x].key < 0) {
		pRet.x = -100.0f;
		pRet.y = -100.0f;
		return pRet;
	}
	if (output != nullptr) {
		if (biInter) {
			PPoint pBi;
			pBi.x = pRet.x - ((int)pRet.x);
			pBi.y = pRet.y - ((int)pRet.y);
			int xside = pRet.x >= w - 1;
			int yside = pRet.y >= h - 1;
			Bilinear(pBi, &footage[((int)pRet.y)*w + (int)pRet.x], &footage[((int)pRet.y + 1 - yside)*w + (int)pRet.x], &footage[((int)pRet.y)*w + (int)pRet.x + 1 - xside], &footage[((int)pRet.y + 1 - yside)*w + (int)pRet.x + 1 - xside], output);
		}
		else {
			output->r = footage[((int)pRet.y)*w + (int)pRet.x].r;
			output->g = footage[((int)pRet.y)*w + (int)pRet.x].g;
			output->b = footage[((int)pRet.y)*w + (int)pRet.x].b;
			output->a = footage[((int)pRet.y)*w + (int)pRet.x].a;
			output->key = footage[((int)pRet.y)*w + (int)pRet.x].key;
		}
	}
	return pRet;
}



typedef struct cpuCalcKeyItData {
	float (*CalcKey)(float minThres, float maxThres, Pixel &p);
	Pixel* footage;
	int w, h;
	float minThres, maxThres;
}cpuCalcKeyItData;

typedef struct cpuSortItData {
	Pixel* footage;
	PixelSortPatternParm* param;
	PPoint (*traverseF)(int w, int h, PPoint now, Pixel* footage, bool previous, bool biInter, PixelSortPatternParm* param, Pixel* output);
	int (*maxLengthF)(int w, int h, PPoint now, PixelSortPatternParm* param);
	bool bilinear;
	int w, h;
	bool reverse_sort_order;
	bool sort_alpha;
}cpuSortItData;

PF_Err iterate8_toCPU(void* refcon, A_long x, A_long y, PF_Pixel *in, PF_Pixel *out) {
	cpuCalcKeyItData* it = (cpuCalcKeyItData*)refcon;
	if (x < 0 || y < 0 || x >= it->w || y >= it->h) {
		return PF_Err_NONE;
	}
	else {
		(it->footage)[y*it->w + x].r = (float)(in->red);
		(it->footage)[y*it->w + x].g = (float)(in->green);
		(it->footage)[y*it->w + x].b = (float)(in->blue);
		(it->footage)[y*it->w + x].a = (float)(in->alpha);
		(it->CalcKey)(it->minThres, it->maxThres, (it->footage)[y*it->w + x]);
		/*if ((it->footage)[y*it->w + x].key >= 0.0f) {
			(it->footage)[y*it->w + x].r = 255;
			(it->footage)[y*it->w + x].g /= 2;
			(it->footage)[y*it->w + x].b /= 2;
		}*/
	}
	return PF_Err_NONE;
}

//////SEARCHING ALGORITHM
Pixel searchHelperCPU(Pixel* nums, int left, int right, int k);
Pixel CPUkthLargestElement(int k, Pixel* nums, int count) {
	return searchHelperCPU(nums, 0, count - 1, count - k + 1);
}

Pixel searchHelperCPU(Pixel* nums, int left, int right, int k) {
	if (left == right) {
		return nums[left];
	}
	int i = left, j = right;
	Pixel pivot = nums[(i + j) / 2];
	while (i <= j) {
		while (i <= j && nums[i].key < pivot.key) {
			i++;
		}
		while (i <= j && nums[j].key > pivot.key) {
			j--;
		}
		if (i <= j) {
			Pixel temp = nums[i];
			nums[i] = nums[j];
			nums[j] = temp;
			i++;
			j--;
		}
	}
	if (left + k - 1 <= j) {
		return searchHelperCPU(nums, left, j, k);
	}
	if (left + k - 1 < i) {
		return nums[left + k - 1];
	}
	return searchHelperCPU(nums, i, right, k - (i - left));
}


//////
PF_Err iterate8_SortCPU(void* refcon, A_long x, A_long y, PF_Pixel *in, PF_Pixel *out) {
	cpuSortItData* it = (cpuSortItData*)refcon;
	if (x < 0 || y < 0 || x >= it->w || y >= it->h) {
		return PF_Err_NONE;
	}
	if (it->footage[y*it->w + x].key <= 0) {
		out->red = (A_u_char)(it->footage[y*it->w + x]).r;
		out->green = (A_u_char)(it->footage[y*it->w + x]).g;
		out->blue = (A_u_char)(it->footage[y*it->w + x]).b;
		out->alpha = (A_u_char)(it->footage[y*it->w + x]).a;
		return PF_Err_NONE;
	}
	int count = 0;
	PPoint nowp;
	nowp.x = (float)x;
	nowp.y = (float)y;
	while (nowp.x >= 0) {//search next
		count++;
		nowp = it->traverseF(it->w, it->h, nowp, it->footage, false, false, it->param, nullptr);
	}
	nowp.x = (float)x;
	nowp.y = (float)y;
	count--;
	int myIndex = -1;
	while (nowp.x >= 0) {//search prev
		count++;
		myIndex++;
		nowp = it->traverseF(it->w, it->h, nowp, it->footage, true, false, it->param, nullptr);
	}
	int total = count;
	if (total <= 1) {
		out->red = (A_u_char)(it->footage[y*it->w + x]).r;
		out->green = (A_u_char)(it->footage[y*it->w + x]).g;
		out->blue = (A_u_char)(it->footage[y*it->w + x]).b;
		out->alpha = (A_u_char)(it->footage[y*it->w + x]).a;
		return PF_Err_NONE;
	}
	Pixel* pArr = (Pixel*)malloc(sizeof(Pixel)*count);
	for (int i = 0; i < count; i++) {
		pArr[i].key = -100;
	}
	nowp.x = (float)x;
	nowp.y = (float)y;
	while (count>0&&nowp.x >= 0) {//fill next
		nowp = it->traverseF(it->w, it->h, nowp, it->footage, false, it->bilinear, it->param, &pArr[total-count] );
		count--;
	}
	nowp.x = (float)x;
	nowp.y = (float)y;
	nowp = it->traverseF(it->w, it->h, nowp, it->footage, true, false, it->param, nullptr);
	while (count>0 && nowp.x >= 0) {//fill prev
		nowp = it->traverseF(it->w, it->h, nowp, it->footage, true, it->bilinear, it->param, &pArr[total - count]);
		count--;
	}
	//sort pArr
	Pixel result = CPUkthLargestElement(total-myIndex-1, pArr, total);
	//end sort pArr
	if (result.key > 0) {
		out->red = (A_u_char)result.r;
		out->green = (A_u_char)result.g;
		out->blue = (A_u_char)result.b;
	}else {
		out->red = (A_u_char)(it->footage[y*it->w + x]).r;
		out->green = (A_u_char)(it->footage[y*it->w + x]).g;
		out->blue = (A_u_char)(it->footage[y*it->w + x]).b;
	}

	out		->alpha = (A_u_char)(it->footage[y*it->w + x]).a;
	free(pArr);
	return PF_Err_NONE;
}

PF_Err prepareCPUinput(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], Pixel** &CPUinputH) {
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	cpuCalcKeyItData itInput;
	itInput.w = (in_data->width*in_data->downsample_x.num / in_data->downsample_x.den);
	itInput.h = (in_data->height*in_data->downsample_y.num / in_data->downsample_y.den);
	itInput.minThres = (float)(params[UIP_MinThres]->u.fs_d.value);
	itInput.maxThres = (float)(params[UIP_MaxThres]->u.fs_d.value);
	switch (params[UIP_SortBy]->u.pd.value)
	{
	case PSB_Hue:
		itInput.CalcKey = CalcKeyCPUHue;
		break;
	case PSB_Saturation:
		itInput.CalcKey = CalcKeyCPUSat;
		break;
	case PSB_Luminance:
		itInput.CalcKey = CalcKeyCPULum;
		break;
	case PSB_R:
		itInput.CalcKey = CalcKeyCPUR;
		break;
	case PSB_G:
		itInput.CalcKey = CalcKeyCPUG;
		break;
	case PSB_B:
		itInput.CalcKey = CalcKeyCPUB;
		break;
	default:
		break;
	}
	
	CPUinputH = (Pixel**)suites.HandleSuite1()->host_new_handle(sizeof(Pixel) * itInput.w * itInput.h);
	itInput.footage = *CPUinputH;
	//fill the pixels
	suites.Iterate8Suite1()->iterate(in_data, 0, 0, &(params[0]->u.ld), NULL, &itInput, &iterate8_toCPU, &(params[0]->u.ld));
	return PF_Err_NONE;
}

PF_Err PixelSortCPU(PF_InData	 *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef	*output, PixelSortPatternParm *pattern_parm, Pixel** &CPUinputH) {
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	gSuites = &suites;
	cpuSortItData itInput;
	itInput.w = (in_data->width*in_data->downsample_x.num / in_data->downsample_x.den);
	itInput.h = (in_data->height*in_data->downsample_y.num / in_data->downsample_y.den);
	itInput.footage = *CPUinputH;
	itInput.param = pattern_parm;
	switch (pattern_parm->pattern) {
	case PSP_Spiral:
		itInput.traverseF = traverseSpiral;
		break;
	default:
		itInput.traverseF = traverseLinear;
		break;
	}
	itInput.reverse_sort_order = params[UIP_RevSortOrder]->u.bd.value!=0;
	itInput.sort_alpha = params[UIP_SortAlpha]->u.bd.value!=0;
	itInput.bilinear = params[UIP_AntiAliasing]->u.bd.value!=0;
	gSuites->Iterate8Suite1()->iterate(in_data, 0, 0,NULL, NULL, &itInput, &iterate8_SortCPU, output);
	return PF_Err_NONE;
}
