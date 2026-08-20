#!/usr/bin/env python3
"""
Epistemic Pipeline Diagnostics
================================

Checks connectivity and state of the epistemic pipeline components.
Run this to diagnose why the pipeline is stuck in curiosity mode.

Usage:
    python3 epistemic_diagnostics.py
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / "src"))

def check_ollama():
    """Check if Ollama is running and accessible."""
    print("\n" + "="*60)
    print("🔍 OLLAMA CONNECTIVITY CHECK")
    print("="*60)
    
    try:
        import urllib.request
        url = "http://localhost:11434/api/tags"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                print("✅ Ollama service is running")
                print(f"   Response: {resp.status} OK")
                return True
        except Exception as e:
            print(f"❌ Ollama service error: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to check Ollama: {e}")
        return False

def check_neo4j():
    """Check if Neo4j is running and accessible."""
    print("\n" + "="*60)
    print("🔍 NEO4J CONNECTIVITY CHECK")
    print("="*60)
    
    try:
        from neo4j import GraphDatabase
        import os
        
        uri = os.getenv("EPISTEMIC_NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("EPISTEMIC_NEO4J_USER", "neo4j")
        password = os.getenv("EPISTEMIC_NEO4J_PASSWORD", "epistemic123")
        
        print(f"Connecting to: {uri}")
        print(f"User: {user}")
        
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("MATCH (t:Theory) RETURN count(t) as count")
                count = result.single()["count"]
                print(f"✅ Neo4j is running")
                print(f"   Theories in database: {count}")
                driver.close()
                return True
        except Exception as e:
            print(f"❌ Neo4j connection failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to check Neo4j: {e}")
        return False

def check_theories():
    """Check the state of theories in Neo4j."""
    print("\n" + "="*60)
    print("🔍 THEORY DATABASE CHECK")
    print("="*60)
    
    try:
        from neo4j import GraphDatabase
        import os
        
        uri = os.getenv("EPISTEMIC_NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("EPISTEMIC_NEO4J_USER", "neo4j")
        password = os.getenv("EPISTEMIC_NEO4J_PASSWORD", "epistemic123")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # Get all theories
            result = session.run("""
                MATCH (t:Theory)
                RETURN t.id, t.text, t.confidence_score, t.attempts, t.predictive_success_rate
                ORDER BY t.confidence_score DESC
            """)
            
            theories = result.fetch(100)
            
            if not theories:
                print("❌ No theories in database!")
                print("   Run: world_model.bootstrap_axioms()")
            else:
                print(f"✅ Found {len(theories)} theories:")
                print("\n  Theory Summary:")
                print(f"  {'ID':<8} {'Confidence':<12} {'Attempts':<10} {'PSR':<10} {'Text':<40}")
                print("  " + "-"*80)
                
                untested = 0
                weak = 0
                strong = 0
                
                for theory in theories[:10]:  # Show first 10
                    theory_id = theory["t.id"][:8]
                    conf = theory["t.confidence_score"]
                    attempts = theory["t.attempts"]
                    psr = theory["t.predictive_success_rate"]
                    text = theory["t.text"][:40]
                    
                    if attempts == 0:
                        untested += 1
                        status = "untested"
                    elif conf < 0.3:
                        weak += 1
                        status = "weak"
                    elif conf >= 0.5:
                        strong += 1
                        status = "strong"
                    else:
                        status = "testing"
                    
                    print(f"  {theory_id} {conf:>10.2f}  {attempts:>8}  {psr:>8.2f}  {text} [{status}]")
                
                print(f"\n  Stats: {untested} untested, {weak} weak, {strong} strong")
            
            driver.close()
            return len(theories) > 0
    except Exception as e:
        print(f"❌ Failed to check theories: {e}")
        return False

def check_cognition():
    """Check if CognitionClient can be initialized."""
    print("\n" + "="*60)
    print("🔍 COGNITION CLIENT CHECK")
    print("="*60)
    
    try:
        from autonomy.cognition_client import CognitionClient
        print("✅ CognitionClient imported successfully")
        
        try:
            client = CognitionClient()
            print("✅ CognitionClient initialized")
            return True
        except Exception as e:
            print(f"❌ CognitionClient initialization failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to import CognitionClient: {e}")
        return False

def main():
    """Run all diagnostics."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " " * 10 + "EPISTEMIC PIPELINE DIAGNOSTICS" + " " * 18 + "║")
    print("╚" + "="*58 + "╝")
    
    checks = {
        "Ollama": check_ollama(),
        "Neo4j": check_neo4j(),
        "Theories": check_theories(),
        "CognitionClient": check_cognition(),
    }
    
    print("\n" + "="*60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All checks passed! The pipeline should work.")
    else:
        print("❌ Some checks failed. See details above.")
        print("\nCommon fixes:")
        print("  1. Start Ollama:  ollama serve")
        print("  2. Start Neo4j:   docker-compose -f docker-compose.neo4j.yml up -d")
        print("  3. Bootstrap:     python3 -c \"from epistemic.layer1_world_model import WorldModel; w = WorldModel(); w.bootstrap_axioms(); print('Bootstrapped')\"")
    
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
