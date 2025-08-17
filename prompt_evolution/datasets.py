#!/usr/bin/env python3
"""
数据集加载器
支持CPTS和WikiSection等标准文本分割数据集
"""

import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .fitness import SegmentationSample


class DatasetLoader:
    """数据集加载器基类"""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
    
    def load_samples(self, dataset_name: str, split: str = "test", limit: int = None) -> List[SegmentationSample]:
        """加载数据集样本"""
        if dataset_name.startswith("cpts"):
            return self._load_cpts_samples(split, limit)
        elif dataset_name.startswith("wikisection"):
            return self._load_wikisection_samples(dataset_name, split, limit)
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
    
    def _load_cpts_samples(self, split: str = "test", limit: int = None) -> List[SegmentationSample]:
        """加载CPTS数据集"""
        cpts_file = self.data_dir / "CPTS" / "CPTS" / f"{split}.json"
        
        if not cpts_file.exists():
            print(f"⚠️ CPTS数据集文件不存在: {cpts_file}")
            return self._create_dummy_samples()
        
        try:
            with open(cpts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = []
            for item in data[:limit] if limit else data:
                sample = self._parse_cpts_item(item)
                if sample:
                    samples.append(sample)
            
            print(f"✅ 加载CPTS {split}集: {len(samples)} 个样本")
            return samples
            
        except Exception as e:
            print(f"❌ 加载CPTS数据集失败: {e}")
            return self._create_dummy_samples()
    
    def _parse_cpts_item(self, item: Dict[str, Any]) -> Optional[SegmentationSample]:
        """解析CPTS数据项"""
        try:
            # 重构完整文本
            paragraphs = item.get("paragraph_list", [])
            if not paragraphs:
                return None
            
            full_text = ""
            boundaries = []
            topic_labels = []
            
            for para in paragraphs:
                para_text = para.get("text", "").strip()
                topic_index = para.get("topic_index", 0)
                
                if para_text:
                    # 记录边界（当前文本长度）
                    if full_text:  # 不是第一个段落
                        boundaries.append(len(full_text))
                    
                    full_text += para_text
                    topic_labels.append(topic_index)
            
            if not full_text:
                return None
            
            return SegmentationSample(
                text=full_text,
                ground_truth_boundaries=boundaries,
                topic_labels=topic_labels,
                metadata={
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "source": "CPTS"
                }
            )
            
        except Exception as e:
            print(f"解析CPTS项目失败: {e}")
            return None
    
    def _load_wikisection_samples(self, dataset_name: str, split: str = "test", limit: int = None) -> List[SegmentationSample]:
        """加载WikiSection数据集"""
        wiki_file = self.data_dir / f"{dataset_name}_{split}.json"
        
        if not wiki_file.exists():
            print(f"⚠️ WikiSection数据集文件不存在: {wiki_file}")
            return self._create_dummy_samples()
        
        try:
            with open(wiki_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = []
            items = data[:limit] if limit else data
            
            for item in items:
                sample = self._parse_wikisection_item(item)
                if sample:
                    samples.append(sample)
            
            print(f"✅ 加载WikiSection {dataset_name} {split}集: {len(samples)} 个样本")
            return samples
            
        except Exception as e:
            print(f"❌ 加载WikiSection数据集失败: {e}")
            return self._create_dummy_samples()
    
    def _parse_wikisection_item(self, item: Dict[str, Any]) -> Optional[SegmentationSample]:
        """解析WikiSection数据项"""
        try:
            text = item.get("text", "").strip()
            annotations = item.get("annotations", [])
            
            if not text or not annotations:
                return None
            
            # 提取边界位置
            boundaries = []
            section_labels = []
            
            for ann in annotations:
                if ann.get("class") == "SectionAnnotation":
                    begin = ann.get("begin", 0)
                    if begin > 0:
                        boundaries.append(begin)
                    section_labels.append(ann.get("sectionLabel", ""))
            
            # 去重并排序边界
            boundaries = sorted(list(set(boundaries)))
            
            return SegmentationSample(
                text=text,
                ground_truth_boundaries=boundaries,
                topic_labels=None,  # WikiSection没有数值主题标签
                metadata={
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "type": item.get("type", ""),
                    "section_labels": section_labels,
                    "source": "WikiSection"
                }
            )
            
        except Exception as e:
            print(f"解析WikiSection项目失败: {e}")
            return None
    
    def _create_dummy_samples(self) -> List[SegmentationSample]:
        """创建虚拟样本用于测试"""
        print("📝 创建虚拟测试样本")
        
        dummy_samples = [
            # 中文样本
            SegmentationSample(
                text="人工智能技术发展迅速，正在改变各行各业。机器学习是人工智能的核心技术，包括监督学习、无监督学习和强化学习。深度学习作为机器学习的重要分支，在图像识别、自然语言处理等领域取得了突破性进展。",
                ground_truth_boundaries=[22, 65],
                topic_labels=[0, 1, 2],
                metadata={"source": "dummy", "language": "chinese"}
            ),
            
            # 英文样本
            SegmentationSample(
                text="Artificial intelligence is rapidly evolving. Machine learning forms the core of AI technology. Deep learning has achieved breakthrough results in computer vision and natural language processing.",
                ground_truth_boundaries=[42, 87],
                topic_labels=[0, 1, 2], 
                metadata={"source": "dummy", "language": "english"}
            ),
            
            # 长文本样本
            SegmentationSample(
                text="量子计算是一种利用量子力学现象进行信息处理的计算模式。与经典计算机使用比特不同，量子计算机使用量子比特（qubit）进行计算。量子比特可以同时处于0和1的叠加态，这使得量子计算机在某些问题上具有指数级的加速能力。目前，各大科技公司都在投入大量资源研发量子计算技术。IBM、Google、微软等公司已经构建了不同规模的量子计算机原型。然而，量子计算仍面临许多技术挑战，包括量子相干性保持、错误纠正、扩展性等问题。",
                ground_truth_boundaries=[25, 85, 140, 180, 230],
                topic_labels=[0, 0, 1, 2, 2, 3],
                metadata={"source": "dummy", "language": "chinese", "topic": "quantum_computing"}
            ),
            
            # 技术文档样本
            SegmentationSample(
                text="本文档介绍了API的基本使用方法。首先需要获取API密钥，然后配置认证信息。接下来可以调用各种API端点获取数据。最后需要处理API响应和错误处理。",
                ground_truth_boundaries=[18, 38, 62],
                topic_labels=[0, 1, 2, 3],
                metadata={"source": "dummy", "language": "chinese", "genre": "technical"}
            )
        ]
        
        return dummy_samples
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            "available_datasets": [],
            "data_directory": str(self.data_dir),
            "status": {}
        }
        
        # 检查CPTS数据集
        cpts_dir = self.data_dir / "CPTS" / "CPTS"
        if cpts_dir.exists():
            cpts_files = list(cpts_dir.glob("*.json"))
            info["available_datasets"].append("cpts")
            info["status"]["cpts"] = {
                "available": True,
                "files": [f.name for f in cpts_files],
                "location": str(cpts_dir)
            }
        else:
            info["status"]["cpts"] = {"available": False, "reason": "目录不存在"}
        
        # 检查WikiSection数据集
        wiki_files = list(self.data_dir.glob("wikisection_*.json"))
        if wiki_files:
            info["available_datasets"].append("wikisection")
            info["status"]["wikisection"] = {
                "available": True,
                "files": [f.name for f in wiki_files],
                "languages": list(set(f.name.split('_')[1] for f in wiki_files)),
                "domains": list(set(f.name.split('_')[2] for f in wiki_files))
            }
        else:
            info["status"]["wikisection"] = {"available": False, "reason": "文件不存在"}
        
        return info
    
    def sample_for_quick_test(self, count: int = 10) -> List[SegmentationSample]:
        """为快速测试采样数据"""
        all_samples = []
        
        # 尝试加载真实数据集
        try:
            cpts_samples = self.load_samples("cpts", "test", limit=5)
            all_samples.extend(cpts_samples)
        except:
            pass
        
        # 添加虚拟样本
        dummy_samples = self._create_dummy_samples()
        all_samples.extend(dummy_samples)
        
        # 随机采样
        if len(all_samples) > count:
            return random.sample(all_samples, count)
        else:
            return all_samples


class DatasetAnalyzer:
    """数据集分析器"""
    
    @staticmethod
    def analyze_samples(samples: List[SegmentationSample]) -> Dict[str, Any]:
        """分析样本统计信息"""
        if not samples:
            return {"count": 0}
        
        # 基本统计
        text_lengths = [len(sample.text) for sample in samples]
        boundary_counts = [len(sample.ground_truth_boundaries) for sample in samples]
        
        # 语言分布
        languages = {}
        sources = {}
        
        for sample in samples:
            metadata = sample.metadata or {}
            lang = metadata.get("language", "unknown")
            source = metadata.get("source", "unknown")
            
            languages[lang] = languages.get(lang, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "count": len(samples),
            "text_length": {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "avg": sum(text_lengths) / len(text_lengths),
                "total": sum(text_lengths)
            },
            "boundaries": {
                "min": min(boundary_counts),
                "max": max(boundary_counts),
                "avg": sum(boundary_counts) / len(boundary_counts),
                "total": sum(boundary_counts)
            },
            "languages": languages,
            "sources": sources
        }


if __name__ == "__main__":
    # 测试数据集加载器
    print("📊 测试数据集加载器")
    
    loader = DatasetLoader()
    
    # 获取数据集信息
    info = loader.get_dataset_info()
    print(f"可用数据集: {info['available_datasets']}")
    
    # 加载测试样本
    samples = loader.sample_for_quick_test(5)
    print(f"加载样本数量: {len(samples)}")
    
    # 分析样本
    analyzer = DatasetAnalyzer()
    analysis = analyzer.analyze_samples(samples)
    print(f"样本分析: {analysis}")
    
    print("✅ 数据集加载器测试完成")