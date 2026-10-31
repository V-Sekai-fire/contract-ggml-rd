struct Slot {
  uint base;
  uint pad0;
  uint pad1;
  uint pad2;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> params;
[[vk::binding(1, 0)]]
RWStructuredBuffer<float> s0;
[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> s1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> s2;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> dst;
[[vk::binding(0, 1)]]
ConstantBuffer<Slot> slot;

uint pw(uint k) {
  return params[(slot.base + k)];
}

void unravel4(uint e, uint w, out uint i0, out uint i1, out uint i2, out uint i3) {
  uint n0 = pw(w);
  uint n1 = pw((w + 1u));
  uint n2 = pw((w + 2u));
  i0 = (e % n0);
  uint r = (e / n0);
  i1 = (r % n1);
  r = (r / n1);
  i2 = (r % n2);
  i3 = (r / n2);
}

uint off4(uint w, uint i0, uint i1, uint i2, uint i3) {
  return ((((pw((w + 8u)) + (i0 * pw((w + 4u)))) + (i1 * pw((w + 5u)))) + (i2 * pw((w + 6u)))) + (i3 * pw((w + 7u))));
}

float sqrt_exact(float x) {
  if ((!((x >= 5.421011e-20f) && (x <= 3.4028235e38f)))) {
    return sqrt(x);
  }
  precise float y0 = rsqrt(x);
  precise float g0 = (x * y0);
  precise float h0 = (0.5f * y0);
  precise float r0 = fma((-g0), g0, x);
  precise float g1 = fma(h0, r0, g0);
  precise float r1 = fma((-g1), g1, x);
  precise float g2 = fma(h0, r1, g1);
  uint bits = asuint(g2);
  precise float u_up = asfloat(((bits & 2139095040u) - 192937984u));
  precise float u_dn = (((bits & 8388607u) == 0u) ? (0.5f * u_up) : u_up);
  precise float e = fma((-g2), g2, x);
  precise float up = (e - (g2 * u_up));
  precise float dn = (e + (g2 * u_dn));
  precise float tu = ((0.25f * u_up) * u_up);
  precise float td = ((0.25f * u_dn) * u_dn);
  if ((up >= tu)) {
    return asfloat((bits + 1u));
  }
  if ((dn < td)) {
    return asfloat((bits - 1u));
  }
  return g2;
}

float div_exact(float a, float b) {
  float aa = abs(a);
  float bb = abs(b);
  float qh = (a / b);
  float aq = abs(qh);
  if ((!((((aa >= 7.888609e-31f) && (aa <= 1.2676506e30f)) && ((bb >= 7.888609e-31f) && (bb <= 1.2676506e30f))) && ((aq >= 7.888609e-31f) && (aq <= 1.2676506e30f))))) {
    return qh;
  }
  precise float y0 = (1.0f / bb);
  precise float e = fma((-bb), y0, 1.0f);
  precise float y1 = fma(y0, e, y0);
  precise float q0 = (aa * y1);
  precise float r = fma((-bb), q0, aa);
  precise float q1 = fma(r, y1, q0);
  uint bits = asuint(q1);
  precise float u_up = asfloat(((bits & 2139095040u) - 192937984u));
  precise float u_dn = (((bits & 8388607u) == 0u) ? (0.5f * u_up) : u_up);
  precise float r2 = fma((-q1), bb, aa);
  precise float tu = ((0.5f * u_up) * bb);
  precise float td = ((0.5f * u_dn) * bb);
  uint out = bits;
  if ((r2 >= tu)) {
    out = (bits + 1u);
  } else {
    if ((r2 < (-td))) {
      out = (bits - 1u);
    }
  }
  return asfloat((out | ((asuint(a) ^ asuint(b)) & 2147483648u)));
}

uint zen2_word(uint w) {
  if ((w < 256u)) {
    if ((w < 128u)) {
      if ((w < 64u)) {
        if ((w < 32u)) {
          if ((w < 16u)) {
            if ((w < 8u)) {
              if ((w < 4u)) {
                if ((w < 2u)) {
                  if ((w < 1u)) {
                    return 1431633920u;
                  } else {
                    return 1431633920u;
                  }
                } else {
                  if ((w < 3u)) {
                    return 1431633920u;
                  } else {
                    return 2863289685u;
                  }
                }
              } else {
                if ((w < 6u)) {
                  if ((w < 5u)) {
                    return 1431983445u;
                  } else {
                    return 1431633920u;
                  }
                } else {
                  if ((w < 7u)) {
                    return 1431983445u;
                  } else {
                    return 2863289685u;
                  }
                }
              }
            } else {
              if ((w < 12u)) {
                if ((w < 10u)) {
                  if ((w < 9u)) {
                    return 1431633925u;
                  } else {
                    return 1431635285u;
                  }
                } else {
                  if ((w < 11u)) {
                    return 1431655765u;
                  } else {
                    return 1431655765u;
                  }
                }
              } else {
                if ((w < 14u)) {
                  if ((w < 13u)) {
                    return 1431639381u;
                  } else {
                    return 1431635285u;
                  }
                } else {
                  if ((w < 15u)) {
                    return 357892117u;
                  } else {
                    return 1433032021u;
                  }
                }
              }
            }
          } else {
            if ((w < 24u)) {
              if ((w < 20u)) {
                if ((w < 18u)) {
                  if ((w < 17u)) {
                    return 2857724501u;
                  } else {
                    return 1431655685u;
                  }
                } else {
                  if ((w < 19u)) {
                    return 2857724501u;
                  } else {
                    return 1426412801u;
                  }
                }
              } else {
                if ((w < 22u)) {
                  if ((w < 21u)) {
                    return 1448432981u;
                  } else {
                    return 1515543125u;
                  }
                } else {
                  if ((w < 23u)) {
                    return 1783982677u;
                  } else {
                    return 2857740885u;
                  }
                }
              }
            } else {
              if ((w < 28u)) {
                if ((w < 26u)) {
                  if ((w < 25u)) {
                    return 1783982677u;
                  } else {
                    return 1515547221u;
                  }
                } else {
                  if ((w < 27u)) {
                    return 1448434261u;
                  } else {
                    return 1427461461u;
                  }
                }
              } else {
                if ((w < 30u)) {
                  if ((w < 29u)) {
                    return 1426150661u;
                  } else {
                    return 1448434261u;
                  }
                } else {
                  if ((w < 31u)) {
                    return 1426412885u;
                  } else {
                    return 1448438357u;
                  }
                }
              }
            }
          }
        } else {
          if ((w < 48u)) {
            if ((w < 40u)) {
              if ((w < 36u)) {
                if ((w < 34u)) {
                  if ((w < 33u)) {
                    return 1700156757u;
                  } else {
                    return 2775209621u;
                  }
                } else {
                  if ((w < 35u)) {
                    return 1364530497u;
                  } else {
                    return 1431655749u;
                  }
                }
              } else {
                if ((w < 38u)) {
                  if ((w < 37u)) {
                    return 1431721301u;
                  } else {
                    return 1700156821u;
                  }
                } else {
                  if ((w < 39u)) {
                    return 1431721365u;
                  } else {
                    return 1431721365u;
                  }
                }
              }
            } else {
              if ((w < 44u)) {
                if ((w < 42u)) {
                  if ((w < 41u)) {
                    return 1364546901u;
                  } else {
                    return 1347753285u;
                  }
                } else {
                  if ((w < 43u)) {
                    return 2775210646u;
                  } else {
                    return 1431721365u;
                  }
                }
              } else {
                if ((w < 46u)) {
                  if ((w < 45u)) {
                    return 1347753301u;
                  } else {
                    return 1431983766u;
                  }
                } else {
                  if ((w < 47u)) {
                    return 1347769685u;
                  } else {
                    return 1431721621u;
                  }
                }
              }
            }
          } else {
            if ((w < 56u)) {
              if ((w < 52u)) {
                if ((w < 50u)) {
                  if ((w < 49u)) {
                    return 1701143877u;
                  } else {
                    return 2779096469u;
                  }
                } else {
                  if ((w < 51u)) {
                    return 1364283712u;
                  } else {
                    return 1431655745u;
                  }
                }
              } else {
                if ((w < 54u)) {
                  if ((w < 53u)) {
                    return 1431655749u;
                  } else {
                    return 1431655749u;
                  }
                } else {
                  if ((w < 55u)) {
                    return 1431655749u;
                  } else {
                    return 1431655749u;
                  }
                }
              }
            } else {
              if ((w < 60u)) {
                if ((w < 58u)) {
                  if ((w < 57u)) {
                    return 1431655749u;
                  } else {
                    return 1364283713u;
                  }
                } else {
                  if ((w < 59u)) {
                    return 2779096470u;
                  } else {
                    return 1701143893u;
                  }
                }
              } else {
                if ((w < 62u)) {
                  if ((w < 61u)) {
                    return 1364546885u;
                  } else {
                    return 1705354902u;
                  }
                } else {
                  if ((w < 63u)) {
                    return 1431659861u;
                  } else {
                    return 1705354902u;
                  }
                }
              }
            }
          }
        }
      } else {
        if ((w < 96u)) {
          if ((w < 80u)) {
            if ((w < 72u)) {
              if ((w < 68u)) {
                if ((w < 66u)) {
                  if ((w < 65u)) {
                    return 1363477780u;
                  } else {
                    return 1700091221u;
                  }
                } else {
                  if ((w < 67u)) {
                    return 2778094169u;
                  } else {
                    return 1363481876u;
                  }
                }
              } else {
                if ((w < 70u)) {
                  if ((w < 69u)) {
                    return 1431639381u;
                  } else {
                    return 1700091477u;
                  }
                } else {
                  if ((w < 71u)) {
                    return 1704351321u;
                  } else {
                    return 1704352345u;
                  }
                }
              }
            } else {
              if ((w < 76u)) {
                if ((w < 74u)) {
                  if ((w < 73u)) {
                    return 1704352345u;
                  } else {
                    return 1704352345u;
                  }
                } else {
                  if ((w < 75u)) {
                    return 1704351321u;
                  } else {
                    return 1431656025u;
                  }
                }
              } else {
                if ((w < 78u)) {
                  if ((w < 77u)) {
                    return 1431656025u;
                  } else {
                    return 1363481941u;
                  }
                } else {
                  if ((w < 79u)) {
                    return 1704352361u;
                  } else {
                    return 1431656025u;
                  }
                }
              }
            }
          } else {
            if ((w < 88u)) {
              if ((w < 84u)) {
                if ((w < 82u)) {
                  if ((w < 81u)) {
                    return 1431639121u;
                  } else {
                    return 1772509541u;
                  }
                } else {
                  if ((w < 83u)) {
                    return 1499813205u;
                  } else {
                    return 1772509541u;
                  }
                }
              } else {
                if ((w < 86u)) {
                  if ((w < 85u)) {
                    return 1431655765u;
                  } else {
                    return 1504074085u;
                  }
                } else {
                  if ((w < 87u)) {
                    return 357897297u;
                  } else {
                    return 1432704341u;
                  }
                }
              }
            } else {
              if ((w < 92u)) {
                if ((w < 90u)) {
                  if ((w < 89u)) {
                    return 1499879781u;
                  } else {
                    return 1772509606u;
                  }
                } else {
                  if ((w < 91u)) {
                    return 341120081u;
                  } else {
                    return 357897557u;
                  }
                }
              } else {
                if ((w < 94u)) {
                  if ((w < 93u)) {
                    return 1431655765u;
                  } else {
                    return 1431655765u;
                  }
                } else {
                  if ((w < 95u)) {
                    return 1431655765u;
                  } else {
                    return 1431655765u;
                  }
                }
              }
            }
          }
        } else {
          if ((w < 112u)) {
            if ((w < 104u)) {
              if ((w < 100u)) {
                if ((w < 98u)) {
                  if ((w < 97u)) {
                    return 1721394777u;
                  } else {
                    return 1704617561u;
                  }
                } else {
                  if ((w < 99u)) {
                    return 1704355161u;
                  } else {
                    return 1431721301u;
                  }
                }
              } else {
                if ((w < 102u)) {
                  if ((w < 101u)) {
                    return 1364546837u;
                  } else {
                    return 1721394777u;
                  }
                } else {
                  if ((w < 103u)) {
                    return 1435919705u;
                  } else {
                    return 1364546837u;
                  }
                }
              }
            } else {
              if ((w < 108u)) {
                if ((w < 106u)) {
                  if ((w < 105u)) {
                    return 1704617561u;
                  } else {
                    return 1431721301u;
                  }
                } else {
                  if ((w < 107u)) {
                    return 1704617561u;
                  } else {
                    return 1364612437u;
                  }
                }
              } else {
                if ((w < 110u)) {
                  if ((w < 109u)) {
                    return 1436182105u;
                  } else {
                    return 290805013u;
                  }
                } else {
                  if ((w < 111u)) {
                    return 1431725401u;
                  } else {
                    return 1704617577u;
                  }
                }
              }
            }
          } else {
            if ((w < 120u)) {
              if ((w < 116u)) {
                if ((w < 114u)) {
                  if ((w < 113u)) {
                    return 1448434005u;
                  } else {
                    return 2523290009u;
                  }
                } else {
                  if ((w < 115u)) {
                    return 2594597273u;
                  } else {
                    return 1163203924u;
                  }
                }
              } else {
                if ((w < 118u)) {
                  if ((w < 117u)) {
                    return 1163220309u;
                  } else {
                    return 1431655765u;
                  }
                } else {
                  if ((w < 119u)) {
                    return 1431656789u;
                  } else {
                    return 1448434005u;
                  }
                }
              }
            } else {
              if ((w < 124u)) {
                if ((w < 122u)) {
                  if ((w < 121u)) {
                    return 1431656789u;
                  } else {
                    return 1431656789u;
                  }
                } else {
                  if ((w < 123u)) {
                    return 1163220309u;
                  } else {
                    return 1163203925u;
                  }
                }
              } else {
                if ((w < 126u)) {
                  if ((w < 125u)) {
                    return 1158747476u;
                  } else {
                    return 1449552281u;
                  }
                } else {
                  if ((w < 127u)) {
                    return 1449482649u;
                  } else {
                    return 1163221333u;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ((w < 192u)) {
        if ((w < 160u)) {
          if ((w < 144u)) {
            if ((w < 136u)) {
              if ((w < 132u)) {
                if ((w < 130u)) {
                  if ((w < 129u)) {
                    return 2863031910u;
                  } else {
                    return 1717982549u;
                  }
                } else {
                  if ((w < 131u)) {
                    return 1431655761u;
                  } else {
                    return 2791728742u;
                  }
                }
              } else {
                if ((w < 134u)) {
                  if ((w < 133u)) {
                    return 1716868437u;
                  } else {
                    return 2863031910u;
                  }
                } else {
                  if ((w < 135u)) {
                    return 1716868437u;
                  } else {
                    return 2858837606u;
                  }
                }
              }
            } else {
              if ((w < 140u)) {
                if ((w < 138u)) {
                  if ((w < 137u)) {
                    return 1700091221u;
                  } else {
                    return 1717986918u;
                  }
                } else {
                  if ((w < 139u)) {
                    return 1431655761u;
                  } else {
                    return 1717917013u;
                  }
                }
              } else {
                if ((w < 142u)) {
                  if ((w < 141u)) {
                    return 1717986918u;
                  } else {
                    return 1431655697u;
                  }
                } else {
                  if ((w < 143u)) {
                    return 1431655765u;
                  } else {
                    return 1717986901u;
                  }
                }
              }
            }
          } else {
            if ((w < 152u)) {
              if ((w < 148u)) {
                if ((w < 146u)) {
                  if ((w < 145u)) {
                    return 1431655765u;
                  } else {
                    return 1431655765u;
                  }
                } else {
                  if ((w < 147u)) {
                    return 2576980373u;
                  } else {
                    return 2576980377u;
                  }
                }
              } else {
                if ((w < 150u)) {
                  if ((w < 149u)) {
                    return 2576980377u;
                  } else {
                    return 2576980377u;
                  }
                } else {
                  if ((w < 151u)) {
                    return 2576980377u;
                  } else {
                    return 2576980377u;
                  }
                }
              }
            } else {
              if ((w < 156u)) {
                if ((w < 154u)) {
                  if ((w < 153u)) {
                    return 2576980377u;
                  } else {
                    return 2576980377u;
                  }
                } else {
                  if ((w < 155u)) {
                    return 1431673241u;
                  } else {
                    return 1431655765u;
                  }
                }
              } else {
                if ((w < 158u)) {
                  if ((w < 157u)) {
                    return 1431655765u;
                  } else {
                    return 2577050282u;
                  }
                } else {
                  if ((w < 159u)) {
                    return 2576980377u;
                  } else {
                    return 1431935385u;
                  }
                }
              }
            }
          }
        } else {
          if ((w < 176u)) {
            if ((w < 168u)) {
              if ((w < 164u)) {
                if ((w < 162u)) {
                  if ((w < 161u)) {
                    return 1717982549u;
                  } else {
                    return 1431655765u;
                  }
                } else {
                  if ((w < 163u)) {
                    return 1789565337u;
                  } else {
                    return 1717982549u;
                  }
                }
              } else {
                if ((w < 166u)) {
                  if ((w < 165u)) {
                    return 1431651413u;
                  } else {
                    return 1717983641u;
                  }
                } else {
                  if ((w < 167u)) {
                    return 1431721301u;
                  } else {
                    return 1718262169u;
                  }
                }
              }
            } else {
              if ((w < 172u)) {
                if ((w < 170u)) {
                  if ((w < 169u)) {
                    return 1432769877u;
                  } else {
                    return 1718262169u;
                  }
                } else {
                  if ((w < 171u)) {
                    return 1431655765u;
                  } else {
                    return 1717983641u;
                  }
                }
              } else {
                if ((w < 174u)) {
                  if ((w < 173u)) {
                    return 1431651413u;
                  } else {
                    return 1449547093u;
                  }
                } else {
                  if ((w < 175u)) {
                    return 1718262169u;
                  } else {
                    return 1431651669u;
                  }
                }
              }
            }
          } else {
            if ((w < 184u)) {
              if ((w < 180u)) {
                if ((w < 178u)) {
                  if ((w < 177u)) {
                    return 1414878481u;
                  } else {
                    return 1700092245u;
                  }
                } else {
                  if ((w < 179u)) {
                    return 1704565078u;
                  } else {
                    return 2845420134u;
                  }
                }
              } else {
                if ((w < 182u)) {
                  if ((w < 181u)) {
                    return 1414878481u;
                  } else {
                    return 1431655765u;
                  }
                } else {
                  if ((w < 183u)) {
                    return 1700092245u;
                  } else {
                    return 1700370773u;
                  }
                }
              }
            } else {
              if ((w < 188u)) {
                if ((w < 186u)) {
                  if ((w < 185u)) {
                    return 1700370774u;
                  } else {
                    return 1704565094u;
                  }
                } else {
                  if ((w < 187u)) {
                    return 1704565350u;
                  } else {
                    return 1771674214u;
                  }
                }
              } else {
                if ((w < 190u)) {
                  if ((w < 189u)) {
                    return 1771674214u;
                  } else {
                    return 1704565350u;
                  }
                } else {
                  if ((w < 191u)) {
                    return 1704565350u;
                  } else {
                    return 1700370790u;
                  }
                }
              }
            }
          }
        }
      } else {
        if ((w < 224u)) {
          if ((w < 208u)) {
            if ((w < 200u)) {
              if ((w < 196u)) {
                if ((w < 194u)) {
                  if ((w < 193u)) {
                    return 1431917909u;
                  } else {
                    return 1431655749u;
                  }
                } else {
                  if ((w < 195u)) {
                    return 1163203908u;
                  } else {
                    return 2578081433u;
                  }
                }
              } else {
                if ((w < 198u)) {
                  if ((w < 197u)) {
                    return 1503290969u;
                  } else {
                    return 1436112469u;
                  }
                } else {
                  if ((w < 199u)) {
                    return 1431917909u;
                  } else {
                    return 1163203908u;
                  }
                }
              }
            } else {
              if ((w < 204u)) {
                if ((w < 202u)) {
                  if ((w < 201u)) {
                    return 2577032793u;
                  } else {
                    return 1436112469u;
                  }
                } else {
                  if ((w < 203u)) {
                    return 1163220293u;
                  } else {
                    return 2577032857u;
                  }
                }
              } else {
                if ((w < 206u)) {
                  if ((w < 205u)) {
                    return 1436112469u;
                  } else {
                    return 1163203909u;
                  }
                } else {
                  if ((w < 207u)) {
                    return 1503225433u;
                  } else {
                    return 1431655765u;
                  }
                }
              }
            }
          } else {
            if ((w < 216u)) {
              if ((w < 212u)) {
                if ((w < 210u)) {
                  if ((w < 209u)) {
                    return 1771464089u;
                  } else {
                    return 1700091221u;
                  }
                } else {
                  if ((w < 211u)) {
                    return 2845205913u;
                  } else {
                    return 1700091221u;
                  }
                }
              } else {
                if ((w < 214u)) {
                  if ((w < 213u)) {
                    return 1771464089u;
                  } else {
                    return 1431393365u;
                  }
                } else {
                  if ((w < 215u)) {
                    return 1771464089u;
                  } else {
                    return 1431393365u;
                  }
                }
              }
            } else {
              if ((w < 220u)) {
                if ((w < 218u)) {
                  if ((w < 217u)) {
                    return 1704289625u;
                  } else {
                    return 1771464090u;
                  }
                } else {
                  if ((w < 219u)) {
                    return 1431393365u;
                  } else {
                    return 1704289625u;
                  }
                }
              } else {
                if ((w < 222u)) {
                  if ((w < 221u)) {
                    return 1771464090u;
                  } else {
                    return 1431393365u;
                  }
                } else {
                  if ((w < 223u)) {
                    return 1700091221u;
                  } else {
                    return 1704289689u;
                  }
                }
              }
            }
          }
        } else {
          if ((w < 240u)) {
            if ((w < 232u)) {
              if ((w < 228u)) {
                if ((w < 226u)) {
                  if ((w < 225u)) {
                    return 340071749u;
                  } else {
                    return 1414878549u;
                  }
                } else {
                  if ((w < 227u)) {
                    return 1431918165u;
                  } else {
                    return 1700357721u;
                  }
                }
              } else {
                if ((w < 230u)) {
                  if ((w < 229u)) {
                    return 1704552089u;
                  } else {
                    return 1704617626u;
                  }
                } else {
                  if ((w < 231u)) {
                    return 1771727514u;
                  } else {
                    return 341120341u;
                  }
                }
              }
            } else {
              if ((w < 236u)) {
                if ((w < 234u)) {
                  if ((w < 233u)) {
                    return 1414862165u;
                  } else {
                    return 1414878549u;
                  }
                } else {
                  if ((w < 235u)) {
                    return 1414878549u;
                  } else {
                    return 1431655765u;
                  }
                }
              } else {
                if ((w < 238u)) {
                  if ((w < 237u)) {
                    return 1431656021u;
                  } else {
                    return 1414878549u;
                  }
                } else {
                  if ((w < 239u)) {
                    return 1414878549u;
                  } else {
                    return 1414878549u;
                  }
                }
              }
            }
          } else {
            if ((w < 248u)) {
              if ((w < 244u)) {
                if ((w < 242u)) {
                  if ((w < 241u)) {
                    return 1499814245u;
                  } else {
                    return 1498765669u;
                  }
                } else {
                  if ((w < 243u)) {
                    return 1431656789u;
                  } else {
                    return 1431655765u;
                  }
                }
              } else {
                if ((w < 246u)) {
                  if ((w < 245u)) {
                    return 1427395924u;
                  } else {
                    return 2590616169u;
                  }
                } else {
                  if ((w < 247u)) {
                    return 2590349929u;
                  } else {
                    return 1499830629u;
                  }
                }
              }
            } else {
              if ((w < 252u)) {
                if ((w < 250u)) {
                  if ((w < 249u)) {
                    return 1431656789u;
                  } else {
                    return 1431590229u;
                  }
                } else {
                  if ((w < 251u)) {
                    return 2590612073u;
                  } else {
                    return 1499830885u;
                  }
                }
              } else {
                if ((w < 254u)) {
                  if ((w < 253u)) {
                    return 1431656805u;
                  } else {
                    return 1427395925u;
                  }
                } else {
                  if ((w < 255u)) {
                    return 2573572713u;
                  } else {
                    return 1431656805u;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if ((w < 384u)) {
      if ((w < 320u)) {
        if ((w < 288u)) {
          if ((w < 272u)) {
            if ((w < 264u)) {
              if ((w < 260u)) {
                if ((w < 258u)) {
                  if ((w < 257u)) {
                    return 1700091217u;
                  } else {
                    return 2778093909u;
                  }
                } else {
                  if ((w < 259u)) {
                    return 2795133285u;
                  } else {
                    return 1363481616u;
                  }
                }
              } else {
                if ((w < 262u)) {
                  if ((w < 261u)) {
                    return 1363481680u;
                  } else {
                    return 2795137445u;
                  }
                } else {
                  if ((w < 263u)) {
                    return 2795137381u;
                  } else {
                    return 1704352101u;
                  }
                }
              }
            } else {
              if ((w < 268u)) {
                if ((w < 266u)) {
                  if ((w < 265u)) {
                    return 1704352101u;
                  } else {
                    return 1431655765u;
                  }
                } else {
                  if ((w < 267u)) {
                    return 2795137446u;
                  } else {
                    return 1704352101u;
                  }
                }
              } else {
                if ((w < 270u)) {
                  if ((w < 269u)) {
                    return 1363481941u;
                  } else {
                    return 1704352101u;
                  }
                } else {
                  if ((w < 271u)) {
                    return 1363481685u;
                  } else {
                    return 1431656805u;
                  }
                }
              }
            }
          } else {
            if ((w < 280u)) {
              if ((w < 276u)) {
                if ((w < 274u)) {
                  if ((w < 273u)) {
                    return 1431659861u;
                  } else {
                    return 2522441305u;
                  }
                } else {
                  if ((w < 275u)) {
                    return 2523506265u;
                  } else {
                    return 1158960404u;
                  }
                }
              } else {
                if ((w < 278u)) {
                  if ((w < 277u)) {
                    return 1159025941u;
                  } else {
                    return 1159025941u;
                  }
                } else {
                  if ((w < 279u)) {
                    return 1159025941u;
                  } else {
                    return 1159025941u;
                  }
                }
              }
            } else {
              if ((w < 284u)) {
                if ((w < 282u)) {
                  if ((w < 281u)) {
                    return 1158960405u;
                  } else {
                    return 2523506281u;
                  }
                } else {
                  if ((w < 283u)) {
                    return 1448699481u;
                  } else {
                    return 1163224409u;
                  }
                }
              } else {
                if ((w < 286u)) {
                  if ((w < 285u)) {
                    return 1091851541u;
                  } else {
                    return 1448715881u;
                  }
                } else {
                  if ((w < 287u)) {
                    return 1159026009u;
                  } else {
                    return 1448715881u;
                  }
                }
              }
            }
          }
        } else {
          if ((w < 304u)) {
            if ((w < 296u)) {
              if ((w < 292u)) {
                if ((w < 290u)) {
                  if ((w < 289u)) {
                    return 2572576089u;
                  } else {
                    return 1413828884u;
                  }
                } else {
                  if ((w < 291u)) {
                    return 2505463125u;
                  } else {
                    return 2576770649u;
                  }
                }
              } else {
                if ((w < 294u)) {
                  if ((w < 293u)) {
                    return 1413828884u;
                  } else {
                    return 1431655701u;
                  }
                } else {
                  if ((w < 295u)) {
                    return 2505463129u;
                  } else {
                    return 2572576089u;
                  }
                }
              }
            } else {
              if ((w < 300u)) {
                if ((w < 298u)) {
                  if ((w < 297u)) {
                    return 2576770649u;
                  } else {
                    return 2576770649u;
                  }
                } else {
                  if ((w < 299u)) {
                    return 2576770649u;
                  } else {
                    return 2572576345u;
                  }
                }
              } else {
                if ((w < 302u)) {
                  if ((w < 301u)) {
                    return 2572576345u;
                  } else {
                    return 1431725401u;
                  }
                } else {
                  if ((w < 303u)) {
                    return 1431655769u;
                  } else {
                    return 1413829909u;
                  }
                }
              }
            }
          } else {
            if ((w < 312u)) {
              if ((w < 308u)) {
                if ((w < 306u)) {
                  if ((w < 305u)) {
                    return 1431638293u;
                  } else {
                    return 1717986649u;
                  }
                } else {
                  if ((w < 307u)) {
                    return 1700091225u;
                  } else {
                    return 1431376149u;
                  }
                }
              } else {
                if ((w < 310u)) {
                  if ((w < 309u)) {
                    return 1717982553u;
                  } else {
                    return 1431654677u;
                  }
                } else {
                  if ((w < 311u)) {
                    return 1717982553u;
                  } else {
                    return 1431638293u;
                  }
                }
              }
            } else {
              if ((w < 316u)) {
                if ((w < 314u)) {
                  if ((w < 313u)) {
                    return 1700091225u;
                  } else {
                    return 1717986922u;
                  }
                } else {
                  if ((w < 315u)) {
                    return 1431655701u;
                  } else {
                    return 1431655769u;
                  }
                }
              } else {
                if ((w < 318u)) {
                  if ((w < 317u)) {
                    return 1717986921u;
                  } else {
                    return 286331157u;
                  }
                } else {
                  if ((w < 319u)) {
                    return 1431638293u;
                  } else {
                    return 1431655769u;
                  }
                }
              }
            }
          }
        }
      } else {
        if ((w < 352u)) {
          if ((w < 336u)) {
            if ((w < 328u)) {
              if ((w < 324u)) {
                if ((w < 322u)) {
                  if ((w < 321u)) {
                    return 2576980377u;
                  } else {
                    return 2576980377u;
                  }
                } else {
                  if ((w < 323u)) {
                    return 2576980377u;
                  } else {
                    return 2576980377u;
                  }
                }
              } else {
                if ((w < 326u)) {
                  if ((w < 325u)) {
                    return 2576980377u;
                  } else {
                    return 2576980377u;
                  }
                } else {
                  if ((w < 327u)) {
                    return 1431673241u;
                  } else {
                    return 1431655765u;
                  }
                }
              }
            } else {
              if ((w < 332u)) {
                if ((w < 330u)) {
                  if ((w < 329u)) {
                    return 1431655765u;
                  } else {
                    return 1145324613u;
                  }
                } else {
                  if ((w < 331u)) {
                    return 2576980377u;
                  } else {
                    return 1431655765u;
                  }
                }
              } else {
                if ((w < 334u)) {
                  if ((w < 333u)) {
                    return 1145328981u;
                  } else {
                    return 1436129689u;
                  }
                } else {
                  if ((w < 335u)) {
                    return 1431655765u;
                  } else {
                    return 2576980377u;
                  }
                }
              }
            }
          } else {
            if ((w < 344u)) {
              if ((w < 340u)) {
                if ((w < 338u)) {
                  if ((w < 337u)) {
                    return 1700091217u;
                  } else {
                    return 2845415830u;
                  }
                } else {
                  if ((w < 339u)) {
                    return 1700091217u;
                  } else {
                    return 1704565141u;
                  }
                }
              } else {
                if ((w < 342u)) {
                  if ((w < 341u)) {
                    return 1413829969u;
                  } else {
                    return 1700092309u;
                  }
                } else {
                  if ((w < 343u)) {
                    return 1771674006u;
                  } else {
                    return 1413829969u;
                  }
                }
              }
            } else {
              if ((w < 348u)) {
                if ((w < 346u)) {
                  if ((w < 345u)) {
                    return 1431655829u;
                  } else {
                    return 1700108693u;
                  }
                } else {
                  if ((w < 347u)) {
                    return 1704565158u;
                  } else {
                    return 1413760337u;
                  }
                }
              } else {
                if ((w < 350u)) {
                  if ((w < 349u)) {
                    return 1413829969u;
                  } else {
                    return 1414878549u;
                  }
                } else {
                  if ((w < 351u)) {
                    return 1414878549u;
                  } else {
                    return 1431655829u;
                  }
                }
              }
            }
          }
        } else {
          if ((w < 368u)) {
            if ((w < 360u)) {
              if ((w < 356u)) {
                if ((w < 354u)) {
                  if ((w < 353u)) {
                    return 2590401945u;
                  } else {
                    return 2573624729u;
                  }
                } else {
                  if ((w < 355u)) {
                    return 2573624729u;
                  } else {
                    return 2572576089u;
                  }
                }
              } else {
                if ((w < 358u)) {
                  if ((w < 357u)) {
                    return 2572576089u;
                  } else {
                    return 1498768725u;
                  }
                } else {
                  if ((w < 359u)) {
                    return 1431655765u;
                  } else {
                    return 1427199061u;
                  }
                }
              }
            } else {
              if ((w < 364u)) {
                if ((w < 362u)) {
                  if ((w < 361u)) {
                    return 2590402969u;
                  } else {
                    return 2572576089u;
                  }
                } else {
                  if ((w < 363u)) {
                    return 1431655765u;
                  } else {
                    return 1427199061u;
                  }
                }
              } else {
                if ((w < 366u)) {
                  if ((w < 365u)) {
                    return 2573624729u;
                  } else {
                    return 1431659861u;
                  }
                } else {
                  if ((w < 367u)) {
                    return 1158763589u;
                  } else {
                    return 1498834265u;
                  }
                }
              }
            }
          } else {
            if ((w < 376u)) {
              if ((w < 372u)) {
                if ((w < 370u)) {
                  if ((w < 369u)) {
                    return 1431655509u;
                  } else {
                    return 1503028633u;
                  }
                } else {
                  if ((w < 371u)) {
                    return 1431655765u;
                  } else {
                    return 1503028633u;
                  }
                }
              } else {
                if ((w < 374u)) {
                  if ((w < 373u)) {
                    return 1431393365u;
                  } else {
                    return 1503028633u;
                  }
                } else {
                  if ((w < 375u)) {
                    return 1162957909u;
                  } else {
                    return 1435854169u;
                  }
                }
              }
            } else {
              if ((w < 380u)) {
                if ((w < 378u)) {
                  if ((w < 377u)) {
                    return 1503028633u;
                  } else {
                    return 1162957909u;
                  }
                } else {
                  if ((w < 379u)) {
                    return 1431659865u;
                  } else {
                    return 1503028633u;
                  }
                }
              } else {
                if ((w < 382u)) {
                  if ((w < 381u)) {
                    return 1146164293u;
                  } else {
                    return 1162958165u;
                  }
                } else {
                  if ((w < 383u)) {
                    return 1431659865u;
                  } else {
                    return 1435854233u;
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if ((w < 448u)) {
        if ((w < 416u)) {
          if ((w < 400u)) {
            if ((w < 392u)) {
              if ((w < 388u)) {
                if ((w < 386u)) {
                  if ((w < 385u)) {
                    return 1431654741u;
                  } else {
                    return 1432704341u;
                  }
                } else {
                  if ((w < 387u)) {
                    return 2506446181u;
                  } else {
                    return 2506462565u;
                  }
                }
              } else {
                if ((w < 390u)) {
                  if ((w < 389u)) {
                    return 2523239781u;
                  } else {
                    return 2523240037u;
                  }
                } else {
                  if ((w < 391u)) {
                    return 2523240037u;
                  } else {
                    return 2523240037u;
                  }
                }
              }
            } else {
              if ((w < 396u)) {
                if ((w < 394u)) {
                  if ((w < 393u)) {
                    return 2523240037u;
                  } else {
                    return 2506462565u;
                  }
                } else {
                  if ((w < 395u)) {
                    return 1432720741u;
                  } else {
                    return 1432704357u;
                  }
                }
              } else {
                if ((w < 398u)) {
                  if ((w < 397u)) {
                    return 1431655765u;
                  } else {
                    return 1364545877u;
                  }
                } else {
                  if ((w < 399u)) {
                    return 1360286036u;
                  } else {
                    return 2523502185u;
                  }
                }
              }
            }
          } else {
            if ((w < 408u)) {
              if ((w < 404u)) {
                if ((w < 402u)) {
                  if ((w < 401u)) {
                    return 2573571669u;
                  } else {
                    return 1498764629u;
                  }
                } else {
                  if ((w < 403u)) {
                    return 1431589189u;
                  } else {
                    return 2590615129u;
                  }
                }
              } else {
                if ((w < 406u)) {
                  if ((w < 405u)) {
                    return 1499829845u;
                  } else {
                    return 1431655749u;
                  }
                } else {
                  if ((w < 407u)) {
                    return 1158959428u;
                  } else {
                    return 1499829845u;
                  }
                }
              }
            } else {
              if ((w < 412u)) {
                if ((w < 410u)) {
                  if ((w < 409u)) {
                    return 1431654725u;
                  } else {
                    return 2590348889u;
                  }
                } else {
                  if ((w < 411u)) {
                    return 1431655765u;
                  } else {
                    return 1158959429u;
                  }
                }
              } else {
                if ((w < 414u)) {
                  if ((w < 413u)) {
                    return 1499813205u;
                  } else {
                    return 1158959429u;
                  }
                } else {
                  if ((w < 415u)) {
                    return 1499829589u;
                  } else {
                    return 1158959429u;
                  }
                }
              }
            }
          }
        } else {
          if ((w < 432u)) {
            if ((w < 424u)) {
              if ((w < 420u)) {
                if ((w < 418u)) {
                  if ((w < 417u)) {
                    return 1431655765u;
                  } else {
                    return 2590615194u;
                  }
                } else {
                  if ((w < 419u)) {
                    return 1431655765u;
                  } else {
                    return 1499829849u;
                  }
                }
              } else {
                if ((w < 422u)) {
                  if ((w < 421u)) {
                    return 1158959429u;
                  } else {
                    return 1431655765u;
                  }
                } else {
                  if ((w < 423u)) {
                    return 1499829849u;
                  } else {
                    return 1158959429u;
                  }
                }
              }
            } else {
              if ((w < 428u)) {
                if ((w < 426u)) {
                  if ((w < 425u)) {
                    return 1431655765u;
                  } else {
                    return 1499829849u;
                  }
                } else {
                  if ((w < 427u)) {
                    return 1158959429u;
                  } else {
                    return 1431655765u;
                  }
                }
              } else {
                if ((w < 430u)) {
                  if ((w < 429u)) {
                    return 1431672409u;
                  } else {
                    return 1499829849u;
                  }
                } else {
                  if ((w < 431u)) {
                    return 2590615194u;
                  } else {
                    return 1158959429u;
                  }
                }
              }
            }
          } else {
            if ((w < 440u)) {
              if ((w < 436u)) {
                if ((w < 434u)) {
                  if ((w < 433u)) {
                    return 1430590545u;
                  } else {
                    return 1430590545u;
                  }
                } else {
                  if ((w < 435u)) {
                    return 1430607189u;
                  } else {
                    return 1498764629u;
                  }
                }
              } else {
                if ((w < 438u)) {
                  if ((w < 437u)) {
                    return 1498764630u;
                  } else {
                    return 1498764694u;
                  }
                } else {
                  if ((w < 439u)) {
                    return 1498768790u;
                  } else {
                    return 1499030934u;
                  }
                }
              }
            } else {
              if ((w < 444u)) {
                if ((w < 442u)) {
                  if ((w < 441u)) {
                    return 1499030934u;
                  } else {
                    return 1499030934u;
                  }
                } else {
                  if ((w < 443u)) {
                    return 1499030934u;
                  } else {
                    return 1498768790u;
                  }
                }
              } else {
                if ((w < 446u)) {
                  if ((w < 445u)) {
                    return 1498768790u;
                  } else {
                    return 1498764694u;
                  }
                } else {
                  if ((w < 447u)) {
                    return 1498764630u;
                  } else {
                    return 1431655765u;
                  }
                }
              }
            }
          }
        }
      } else {
        if ((w < 480u)) {
          if ((w < 464u)) {
            if ((w < 456u)) {
              if ((w < 452u)) {
                if ((w < 450u)) {
                  if ((w < 449u)) {
                    return 1363477780u;
                  } else {
                    return 2526697833u;
                  }
                } else {
                  if ((w < 451u)) {
                    return 2522503525u;
                  } else {
                    return 1448695141u;
                  }
                }
              } else {
                if ((w < 454u)) {
                  if ((w < 453u)) {
                    return 1431655765u;
                  } else {
                    return 1364542741u;
                  }
                } else {
                  if ((w < 455u)) {
                    return 1363477780u;
                  } else {
                    return 2522503525u;
                  }
                }
              }
            } else {
              if ((w < 460u)) {
                if ((w < 458u)) {
                  if ((w < 457u)) {
                    return 1431917925u;
                  } else {
                    return 1431651669u;
                  }
                } else {
                  if ((w < 459u)) {
                    return 1363477780u;
                  } else {
                    return 2522437989u;
                  }
                }
              } else {
                if ((w < 462u)) {
                  if ((w < 461u)) {
                    return 1431655765u;
                  } else {
                    return 1363494165u;
                  }
                } else {
                  if ((w < 463u)) {
                    return 2522503525u;
                  } else {
                    return 1431655765u;
                  }
                }
              }
            }
          } else {
            if ((w < 472u)) {
              if ((w < 468u)) {
                if ((w < 466u)) {
                  if ((w < 465u)) {
                    return 1363494164u;
                  } else {
                    return 1448696165u;
                  }
                } else {
                  if ((w < 467u)) {
                    return 1364542805u;
                  } else {
                    return 2522503529u;
                  }
                }
              } else {
                if ((w < 470u)) {
                  if ((w < 469u)) {
                    return 1431655765u;
                  } else {
                    return 1095042324u;
                  }
                } else {
                  if ((w < 471u)) {
                    return 1431917925u;
                  } else {
                    return 1363494165u;
                  }
                }
              }
            } else {
              if ((w < 476u)) {
                if ((w < 474u)) {
                  if ((w < 473u)) {
                    return 1431918949u;
                  } else {
                    return 1363494165u;
                  }
                } else {
                  if ((w < 475u)) {
                    return 1431918949u;
                  } else {
                    return 1363494165u;
                  }
                }
              } else {
                if ((w < 478u)) {
                  if ((w < 477u)) {
                    return 1431917925u;
                  } else {
                    return 1095042325u;
                  }
                } else {
                  if ((w < 479u)) {
                    return 1431655781u;
                  } else {
                    return 2522503529u;
                  }
                }
              }
            }
          }
        } else {
          if ((w < 496u)) {
            if ((w < 488u)) {
              if ((w < 484u)) {
                if ((w < 482u)) {
                  if ((w < 481u)) {
                    return 1431651653u;
                  } else {
                    return 1499027797u;
                  }
                } else {
                  if ((w < 483u)) {
                    return 1162167557u;
                  } else {
                    return 1498764629u;
                  }
                }
              } else {
                if ((w < 486u)) {
                  if ((w < 485u)) {
                    return 1515805017u;
                  } else {
                    return 1162167557u;
                  }
                } else {
                  if ((w < 487u)) {
                    return 1498764629u;
                  } else {
                    return 1515805017u;
                  }
                }
              }
            } else {
              if ((w < 492u)) {
                if ((w < 490u)) {
                  if ((w < 489u)) {
                    return 1162167557u;
                  } else {
                    return 1431655749u;
                  }
                } else {
                  if ((w < 491u)) {
                    return 1499027797u;
                  } else {
                    return 1515870809u;
                  }
                }
              } else {
                if ((w < 494u)) {
                  if ((w < 493u)) {
                    return 1162167621u;
                  } else {
                    return 1431655749u;
                  }
                } else {
                  if ((w < 495u)) {
                    return 1499026773u;
                  } else {
                    return 1515805017u;
                  }
                }
              }
            }
          } else {
            if ((w < 504u)) {
              if ((w < 500u)) {
                if ((w < 498u)) {
                  if ((w < 497u)) {
                    return 1157956869u;
                  } else {
                    return 1162167621u;
                  }
                } else {
                  if ((w < 499u)) {
                    return 1431655749u;
                  } else {
                    return 1431655765u;
                  }
                }
              } else {
                if ((w < 502u)) {
                  if ((w < 501u)) {
                    return 1499027801u;
                  } else {
                    return 1515805017u;
                  }
                } else {
                  if ((w < 503u)) {
                    return 1515870810u;
                  } else {
                    return 84215045u;
                  }
                }
              }
            } else {
              if ((w < 508u)) {
                if ((w < 506u)) {
                  if ((w < 505u)) {
                    return 1162167621u;
                  } else {
                    return 1162167621u;
                  }
                } else {
                  if ((w < 507u)) {
                    return 1162167621u;
                  } else {
                    return 1431655749u;
                  }
                }
              } else {
                if ((w < 510u)) {
                  if ((w < 509u)) {
                    return 1431655765u;
                  } else {
                    return 1431655765u;
                  }
                } else {
                  if ((w < 511u)) {
                    return 1431655765u;
                  } else {
                    return 1431655765u;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

float rsqrt_zen2(float x) {
  uint bits = asuint(x);
  if (isnan(x)) {
    return x;
  }
  uint ex = ((bits >> 23u) & 255u);
  if (((bits >> 31u) != 0u)) {
    return asfloat(((ex == 0u) ? 4286578688u : 4290772992u));
  }
  if ((ex == 0u)) {
    return asfloat(2139095040u);
  }
  if ((bits == 2139095040u)) {
    return 0.0f;
  }
  uint idx = ((bits & 8388607u) >> 11u);
  uint par = (((ex & 1u) == 1u) ? 0u : 1u);
  uint n = ((par * 4096u) + idx);
  uint wd = zen2_word((n >> 4u));
  int res = (int(((wd >> (2u * (n & 15u))) & 3u)) - 1);
  precise float scale = ((par == 1u) ? 2.0f : 1.0f);
  precise float t = (((float(idx) * 2.0f) + 1.0f) * 0.00012207031f);
  precise float xm = ((1.0f + t) * scale);
  precise float q = div_exact(8192.0f, sqrt_exact(xm));
  precise float qr = floor((q + 0.5f));
  int y = (int(qr) + res);
  int sh = ((int(ex) - 127) >> 1u);
  uint eb = ((uint((126 - sh)) << 23u) | (uint((y - 4096)) << 11u));
  precise float r0 = asfloat(eb);
  precise float r2 = (x * r0);
  precise float hr = (r0 * (-0.5f));
  precise float h = fma(r2, r0, (-1.0f));
  return fma(hr, h, r0);
}

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
  uint e = ((((gid.y * pw(54u)) + gid.x) * 256u) + lid.x);
  if ((e >= pw(53u))) {
    return;
  }
  uint i0;
  uint i1;
  uint i2;
  uint i3;
  unravel4(e, 1u, i0, i1, i2, i3);
  uint d = off4(1u, i0, i1, i2, i3);
  uint a = off4(10u, i0, i1, i2, i3);
  float x = s0[a];
  dst[d] = rsqrt_zen2(x);
}
